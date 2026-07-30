import json
import numpy as np
from typing import List, Any, Dict, Tuple
from .schema import FunctionDefinition


def decode_vocabulary(llm: Any) -> Dict[int, str]:
    """Pre-decodifica tutti i token in stringhe Python, ignorando utf-8 invalidi."""
    vocab_path = llm.get_path_to_vocab_file()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    tokens = {}
    for token_str, token_id in vocab.items():
        decoded_str = llm.decode([token_id])
        if '\ufffd' not in decoded_str and decoded_str:
            tokens[token_id] = decoded_str
    return tokens


def build_selection_prompt(user_prompt: str, functions: List[FunctionDefinition]) -> str:
    """Prompt per la selezione del nome della funzione."""
    fn_list = "\n".join(f"- {fn.name}: {fn.description}" for fn in functions)
    return (
        f"<|im_start|>system\n"
        f"You are a function router. Given a user request, reply with ONLY "
        f"the function name that best matches. No explanation.\n"
        f"Available functions:\n{fn_list}\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_extraction_prompt(user_prompt: str, fn: FunctionDefinition) -> str:
    """Prompt per l'estrazione dei parametri di una singola funzione."""
    params_desc = ", ".join(f"{name} ({pdef.type})" for name, pdef in fn.parameters.items())
    return (
        f"<|im_start|>system\n"
        f"Extract the parameters for {fn.name} from the user message.\n"
        f"Parameters: {params_desc}\n"
        f"Reply with ONLY a JSON object of the parameters. "
        f"Extract values exactly as written.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_name_template(functions: List[FunctionDefinition]) -> List[List[Tuple[str, str]]]:
    """Costruisce i template per i nomi delle funzioni (uno per ogni alternativa)."""
    return [[("literal", fn.name)] for fn in functions]


def build_params_template(fn: FunctionDefinition) -> List[Tuple[str, str]]:
    """Costruisce il template lineare per i parametri di una funzione."""
    segments = [("literal", "{")]
    params = list(fn.parameters.items())
    for i, (p_name, p_def) in enumerate(params):
        segments.append(("literal", f'"{p_name}": '))
        t = p_def.type
        kind = t if t in ("number", "integer", "boolean") else "string"
        segments.append((kind, ""))
        if i < len(params) - 1:
            segments.append(("literal", ", "))
    segments.append(("literal", "}"))
    return segments


def matches_template(text: str, template: List[Tuple[str, str]]) -> Tuple[bool, bool]:
    """
    Controlla se `text` è un prefisso valido per il `template`.
    Restituisce (is_valid_prefix, is_finished).
    """
    offset = 0

    for kind, literal_text in template:
        if offset >= len(text):
            return True, False

        if kind == "literal":
            match_len = min(len(text) - offset, len(literal_text))
            if text[offset:offset+match_len] != literal_text[:match_len]:
                return False, False
            offset += match_len
            if match_len < len(literal_text):
                return True, False

        elif kind == "string":
            if text[offset] != '"':
                return False, False
            offset += 1
            special_c = False
            in_string = True

            while offset < len(text) and in_string:
                c = text[offset]
                if special_c:
                    if c not in '"\\/bfnrt':
                        return False, False
                    special_c = False
                    offset += 1
                elif c == '\\':
                    special_c = True
                    offset += 1
                elif c == '"':
                    in_string = False
                    offset += 1
                else:
                    offset += 1

            if in_string:
                return True, False

        elif kind in ("number", "integer"):
            start_offset = offset
            while offset < len(text):
                c = text[offset]
                is_valid = c.isdigit()
                if offset == start_offset and c == '-':
                    is_valid = True
                if kind == "number" and c == '.':
                    is_valid = True
                if not is_valid:
                    break
                offset += 1

            if offset == start_offset:
                return False, False

            # Verifica che un n non sia solo un - 
            # e che non termini con un . (es  "age": 21.)
            if offset < len(text):
                num_str = text[start_offset:offset]
                if num_str == '-' or num_str.endswith('.'):
                    return False, False

        elif kind == "boolean":
            start_offset = offset
            matched_word = None
            for w in ("true", "false"):
                match_len = min(len(text) - offset, len(w))
                if text[offset:offset+match_len] == w[:match_len]:
                    matched_word = w
                    offset += match_len
                    break

            if not matched_word:
                return False, False

            if offset - start_offset < len(matched_word):
                return True, False

    return offset == len(text), offset == len(text)


def matches_any_template(text: str, templates: List[List[Tuple[str, str]]]) -> Tuple[bool, bool]:
    """Applica matches_template a più alternative. Valido se almeno uno lo è."""
    any_valid = False
    any_finished = False
    for tmpl in templates:
        valid, finished = matches_template(text, tmpl)
        if valid:
            any_valid = True
            if finished:
                any_finished = True
    return any_valid, any_finished


def generate_constrained_text(
    llm: Any,
    prompt: str,
    vocab_tokens: Dict[int, str],
    templates: List[List[Tuple[str, str]]],
    max_tokens: int = 100,
) -> str:
    """ 
    Genera testo provando il primo token valido ordinato per logit.
    Se viene respinto dallo shema, si prova il secondo token più probabile.
    Il ciclo continua fino a quando non si trova un token valido.
    """
    generated_text = ""

    for _ in range(max_tokens):
        current_input = llm.encode(prompt + generated_text).tolist()[0]
        logits = llm.get_logits_from_input_ids(current_input)

        # Ordina i logit dal più probabile al meno probabile
        sorted_indices = np.argsort(logits)[::-1]

        best_token_str = None
        is_finished_after_token = False

        for token_id in sorted_indices:
            token_id = int(token_id)
            # Scarta sequenze utf non valide
            if token_id not in vocab_tokens:
                continue

            token_str = vocab_tokens[token_id]
            candidate_text = generated_text + token_str

            is_valid, is_finished = matches_any_template(candidate_text, templates)
            if is_valid:
                best_token_str = token_str
                is_finished_after_token = is_finished
                break

        if best_token_str is None:
            break

        generated_text += best_token_str
        if is_finished_after_token:
            break

    return generated_text


def generate_function_call(
    llm: Any,
    user_prompt: str,
    functions: List[FunctionDefinition],
    vocab_tokens: Dict[int, str],
) -> str:
    """Due fasi: seleziona il nome della funzione, poi ne estrae i parametri."""
    fn_map = {fn.name: fn for fn in functions}

    # Selezione della funzione
    selection_prompt = build_selection_prompt(user_prompt, functions)
    name_alternatives = build_name_template(functions)

    selected_name = generate_constrained_text(
        llm, selection_prompt, vocab_tokens, name_alternatives, max_tokens=30
    )
    selected_name = selected_name.strip()
    selected_fn = fn_map[selected_name]

    # Estrazione dei parametri
    extraction_prompt = build_extraction_prompt(user_prompt, selected_fn)
    params_template = build_params_template(selected_fn)

    params_json = generate_constrained_text(
        llm, extraction_prompt, vocab_tokens, [params_template], max_tokens=100
    )

    return json.dumps({
        "name": selected_name,
        "parameters": json.loads(params_json)
    })
