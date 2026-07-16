import argparse
import json
import os
from llm_sdk import Small_LLM_Model  # type: ignore
from .schema import load_functions_definition, load_prompts, FunctionCallingResult
from .constrained_decoder import Vocabulary
from .generator import generate_constrained


def main() -> None:
    """Punto di ingresso principale: analizza argomenti, carica i dati ed esegue l'LLM."""
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

    print("Initializing model and vocabulary...")
    llm = Small_LLM_Model()
    vocab = Vocabulary(llm)

    results = []
    print(f"Processing {len(prompts)} prompts...")
    for i, test in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] Generating for prompt: {test.prompt}")
        json_str = generate_constrained(llm, test.prompt, functions, vocab)

        try:
            parsed = json.loads(json_str)
            res = FunctionCallingResult(
                prompt=test.prompt,
                name=parsed.get("name", ""),
                parameters=parsed.get("parameters", {})
            )
            results.append(res.model_dump())
        except Exception as e:
            print(f"Error parsing generated JSON: {e}")
            print(f"Generated text was: {json_str}")
            # Fallback in caso di errore
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
