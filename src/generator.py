import math
from typing import List, Any
from .schema import FunctionDefinition
from .constrained_decoder import Vocabulary, JSONStateMachine
from .fsm import build_schema_nfa


def build_prompt(user_prompt: str, functions: List[FunctionDefinition]) -> str:
    prompt = "<|im_start|>system\nYou are a helpful assistant that translates natural language to function calls.\n"
    prompt += "Available functions:\n"
    for fn in functions:
        prompt += f"- {fn.name}: {fn.description}\n"
        for p_name, p_def in fn.parameters.items():
            prompt += f"  - {p_name} ({p_def.type}): {p_def.description or ''}\n"
    prompt += "Output exactly ONE valid JSON object representing the function call.\n"
    prompt += "<|im_end|>\n<|im_start|>user\n"
    prompt += f"{user_prompt}\n<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def generate_constrained(llm: Any, user_prompt: str, functions: List[FunctionDefinition], vocab: Vocabulary) -> str:
    start_state, end_state = build_schema_nfa(functions)
    fsm = JSONStateMachine(start_state, end_state)

    full_prompt = build_prompt(user_prompt, functions)

    # llm.encode returns a 2D tensor, but we need a list of ints for get_logits_from_input_ids
    input_ids_tensor = llm.encode(full_prompt)
    input_ids = input_ids_tensor.tolist()[0]

    generated_ids: List[int] = []

    max_tokens = 100
    for _ in range(max_tokens):
        if fsm.is_finished():
            break

        current_input = input_ids + generated_ids
        logits = llm.get_logits_from_input_ids(current_input)

        valid_tokens = fsm.get_valid_next_tokens(vocab)

        # If no valid tokens, something is wrong, we can just break
        if not valid_tokens:
            break

        # Constrain logits
        constrained_logits = [-math.inf] * len(logits)
        for t_id in valid_tokens:
            if t_id < len(logits):
                constrained_logits[t_id] = logits[t_id]

        # Argmax
        best_token = max(range(len(logits)), key=lambda i: constrained_logits[i])

        generated_ids.append(best_token)

        # Advance the FSM state with the new token's string
        token_str = llm.decode([best_token])
        if not fsm.advance(token_str):
            # This should not happen since we only picked from valid_tokens
            break

    return str(llm.decode(generated_ids))
