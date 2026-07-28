import json
import numpy as np
from typing import List, Any
from .schema import FunctionDefinition
from .constrained_decoder import Vocabulary, JSONStateMachine
from .fsm import build_name_selection_nfa, build_single_function_params_nfa


def build_selection_prompt(user_prompt: str, functions: List[FunctionDefinition]) -> str:
    """Costruisce un prompt per la selezione del nome della funzione."""
    fn_list = "\n".join(
        f"- {fn.name}: {fn.description}" for fn in functions
    )
    prompt = (
        f"<|im_start|>system\n"
        f"You are a function router. Given a user request, reply with ONLY "
        f"the function name that best matches. No explanation.\n"
        f"Available functions:\n{fn_list}\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def build_extraction_prompt(
    user_prompt: str, fn: FunctionDefinition
) -> str:
    """Costruisce un prompt per l'estrazione dei parametri di una singola funzione."""
    params_desc = ", ".join(
        f"{name} ({pdef.type})" for name, pdef in fn.parameters.items()
    )
    prompt = (
        f"<|im_start|>system\n"
        f"Extract the parameters for {fn.name} from the user message.\n"
        f"Parameters: {params_desc}\n"
        f"Reply with ONLY a JSON object of the parameters. "
        f"Extract values exactly as written.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def generate_constrained_text(
    llm: Any,
    prompt: str,
    vocab: Vocabulary,
    fsm: JSONStateMachine,
    max_tokens: int = 100,
) -> str:
    """Genera testo con constrained decoding tramite NFA e Trie del vocabolario."""
    fsm.reset()
    generated_text = ""

    for _ in range(max_tokens):
        if fsm.is_finished():
            break

        current_input = llm.encode(prompt + generated_text).tolist()[0]
        logits = llm.get_logits_from_input_ids(current_input)

        valid_tokens = fsm.get_valid_next_tokens(vocab)
        valid_in_range = [t_id for t_id in valid_tokens if t_id < len(logits)]
        if not valid_in_range:
            break

        logits_np = np.array(logits)
        valid_logits = logits_np[valid_in_range]
        best_idx = np.argmax(valid_logits)
        best_token = valid_in_range[best_idx]

        token_str = llm.decode([best_token])
        generated_text += token_str

        if not fsm.advance(token_str):
            break

    return generated_text


def generate_function_call(
    llm: Any,
    user_prompt: str,
    functions: List[FunctionDefinition],
    vocab: Vocabulary,
) -> str:
    """Orchestratore a due fasi: seleziona la funzione, poi ne estrae i parametri.

    Fase 1: Prompt minimale + NFA ristretto ai nomi → nome della funzione.
    Fase 2: Prompt con singola funzione + NFA per i suoi parametri → JSON parametri.
    Assembla il risultato finale in Python.
    """
    fn_map = {fn.name: fn for fn in functions}

    # --- Fase 1: Selezione della funzione ---
    selection_prompt = build_selection_prompt(user_prompt, functions)
    name_start, name_end = build_name_selection_nfa(functions)
    name_fsm = JSONStateMachine(name_start, name_end)

    selected_name = generate_constrained_text(
        llm, selection_prompt, vocab, name_fsm, max_tokens=30
    )
    selected_name = selected_name.strip()

    selected_fn = fn_map[selected_name]

    # --- Fase 2: Estrazione dei parametri ---
    extraction_prompt = build_extraction_prompt(user_prompt, selected_fn)
    params_start, params_end = build_single_function_params_nfa(selected_fn)
    params_fsm = JSONStateMachine(params_start, params_end)

    params_json = generate_constrained_text(
        llm, extraction_prompt, vocab, params_fsm, max_tokens=100
    )

    result = json.dumps({
        "name": selected_name,
        "parameters": json.loads(params_json)
    })
    return result
