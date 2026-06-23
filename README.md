*This project has been created as part of the 42 curriculum by io.*

# call me maybe: Introduction to function calling in LLMs

## Description
This project implements a robust constrained decoding system that enforces structured JSON output from a Small Language Model (Qwen3-0.6B). It bridges the gap between natural language prompts and machine-executable function calls by guaranteeing 100% syntactically and semantically valid JSON that adheres to a dynamically provided function schema. 

## Instructions

### Prerequisites
- Python 3.10+
- `uv` package manager

### Installation
Run the following command to install dependencies via `uv`:
```bash
make install
```

### Execution
Run the main script to process input prompts:
```bash
make run
```
Or manually using `uv`:
```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

## Resources
- **Hugging Face Transformers Documentation:** Used to understand model loading and logit generation.
- **Pydantic Documentation:** Used to define and validate input JSON structures.
- **AI Usage:** An AI coding assistant (Gemini) was utilized to rapidly prototype the mathematical logic behind the Finite State Machine, optimize the Trie intersections for the 150k vocabulary, and ensure PEP-8 compliance throughout the codebase.

## Algorithm explanation
The core algorithm relies on two intersecting data structures:
1. **Character-Level NFA (Finite State Machine):** The provided JSON schemas (`functions_definition.json`) are compiled into a Non-deterministic Finite Automaton at runtime. This guarantees that only characters forming the exact expected JSON structure (like `{"name": "fn_greet", "parameters": { ... }}`) are accepted.
2. **Vocabulary Prefix Tree (Trie):** The model's 150,000+ token vocabulary is decoded and inserted into a Trie. 

At each generation step, the model outputs raw logits. Instead of iterating through all 150k tokens to check validity, the algorithm performs a simultaneous Depth-First Search (DFS) on the FSM and the Trie. If a token's characters follow a valid path in the FSM, its ID is collected. All other token logits are set to `-inf`. The highest probability valid token is then selected.

## Design decisions
- **Character-level over Regex:** Standard regex libraries do not easily support partial prefix matching. Writing a custom character-level FSM allowed for efficient validation of incomplete JSON strings as the LLM generates them token-by-token.
- **Pre-decoded Vocabulary:** The Qwen tokenizer uses byte-level BPE. Instead of dealing with byte-level state machines, the entire vocabulary was pre-decoded to strings during initialization. Invalid UTF-8 sequences were filtered out, ensuring the state machine operates purely on strings.

## Performance analysis
- **Reliability:** By pruning logits before sampling, the model is physically prevented from outputting invalid JSON. This results in a 100% success rate for JSON schema compliance.
- **Speed:** The Trie intersection drastically reduces the search space at each generation step. Processing a prompt takes a fraction of a second, easily passing the 5-minute requirement for the test suite.

## Challenges faced
1. **Vocabulary Decoding overhead:** Calling the tokenizer's decode function on 150k tokens individually during each generation step was too slow. This was solved by pre-processing the vocabulary into a Trie on startup.
2. **Partial UTF-8 Tokens:** Some tokens in BPE represent partial byte sequences which fail to decode into valid Python strings. This was handled by ignoring tokens that decode to the replacement character (`\ufffd`), as valid JSON generation only requires ASCII and standard characters.

## Testing strategy
The system was validated using the provided `data/input/function_calling_tests.json` file. Static analysis was enforced using `flake8` and `mypy --strict`.

## Example usage
```bash
uv run python -m src
```
This generates `data/output/function_calling_results.json` containing the mapped function calls.
