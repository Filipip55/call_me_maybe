import argparse
import json
import os
from typing import List, Dict, Any
from llm_sdk import Small_LLM_Model  # type: ignore
from .schema import load_functions_definition, load_prompts, FunctionCallingResult, FunctionDefinition
from .constrained_decoder import Vocabulary
from .generator import generate_function_call


def normalize_parameters(fn_name: str, params: Dict[str, Any], functions: List[FunctionDefinition]) -> Dict[str, Any]:
    """Normalizza i tipi dei parametri in base allo schema della funzione."""
    fn_map = {fn.name: fn for fn in functions}
    if fn_name not in fn_map:
        return params
    fn_def = fn_map[fn_name]
    normalized = {}
    for p_name, p_val in params.items():
        if p_name in fn_def.parameters:
            expected_type = fn_def.parameters[p_name].type
            if expected_type == "number":
                try:
                    normalized[p_name] = float(p_val)
                except (ValueError, TypeError):
                    normalized[p_name] = p_val
            elif expected_type == "integer":
                try:
                    normalized[p_name] = int(p_val)
                except (ValueError, TypeError):
                    normalized[p_name] = p_val
            elif expected_type == "boolean":
                if isinstance(p_val, str):
                    normalized[p_name] = p_val.lower() == "true"
                else:
                    normalized[p_name] = bool(p_val)
            else:
                normalized[p_name] = p_val
        else:
            normalized[p_name] = p_val
    return normalized


def main() -> None:
    """Punto di ingresso principale, analizza argomenti ed esegue l'LLM."""
    parser = argparse.ArgumentParser(description="Function Calling via Constrained Decoding")
    parser.add_argument("--functions_definition", type=str, default="data/input/functions_definition.json")
    parser.add_argument("--input", type=str, default="data/input/function_calling_tests.json")
    parser.add_argument("--output", type=str, default="data/output/function_calling_results.json")
    args = parser.parse_args()

    # Crea la cartella di output (se non esiste)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Loading schemas and input data...")
    functions = load_functions_definition(args.functions_definition)
    prompts = load_prompts(args.input)

    if not functions:
        print("Error: No valid function definitions found. Cannot proceed.")
        return

    if not prompts:
        print("No valid prompts to process. Writing empty results file...")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)
        return

    print("Initializing model and vocabulary...")
    llm = Small_LLM_Model()
    vocab = Vocabulary(llm)

    results = []
    print(f"Processing {len(prompts)} prompts...")
    for i, test in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] Generating function call for the following prompt: {test.prompt}")
        json_str = generate_function_call(llm, test.prompt, functions, vocab)

        try:
            parsed = json.loads(json_str)
            fn_name = parsed.get("name", "")
            raw_params = parsed.get("parameters", {})
            norm_params = normalize_parameters(fn_name, raw_params, functions)

            res = FunctionCallingResult(
                prompt=test.prompt,
                name=fn_name,
                parameters=norm_params
            )
            results.append(res.model_dump())
        except Exception as e:
            print(f"Error parsing generated JSON: {e}")
            print(f"Generated text was: {json_str}")
            # Fallback in caso di errore (non fa errori)
            results.append({
                "prompt": test.prompt,
                "name": "error",
                "parameters": {}
            })

    print(f"Writing results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
