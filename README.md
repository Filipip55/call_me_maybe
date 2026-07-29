*This project has been created as part of the 42 curriculum by icoman.*

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
- **AI Usage:** An AI coding assistant (Gemini) was utilized to rapidly prototype the initial complex logic, and subsequently refactor the codebase to achieve extreme simplicity by leveraging NumPy's optimized C backend for sorting, effectively removing over 300 lines of unnecessary logic.

## Algorithm explanation
The core algorithm relies on a highly optimized, stateless **Sorted Logit Scan** combined with linear pattern matching:
1. **Linear Templates:** The provided JSON schemas (`functions_definition.json`) are compiled into a flat list of simple instructions (e.g., `[("literal", '{"name": "'), ("string", ""), ...]`). This acts as the recipe for the valid JSON output.
2. **Sorted Logit Scan:** At each generation step, the model outputs raw logits for its 150,000+ token vocabulary. Instead of building complex trees (Tries) or Finite State Machines, the algorithm uses NumPy's `np.argsort()` to instantly sort all 150k probabilities from highest to lowest.
3. **Stateless Matching:** The algorithm iterates through the sorted tokens. For each token, it concatenates it to the current generated string and runs a purely stateless, ultra-fast Python string comparison against the template recipe. The first token that produces a valid string prefix is accepted.

## Design decisions
- **Stateless Python String Matching over FSMs/Tries:** Managing complex OOP abstractions like NFA nodes, state closures, and Prefix Trees in Python introduced massive overhead and readability issues. A stateless approach reduced the codebase by ~50% while actually slightly improving generation speed.
- **Two-Phase Generation:** The system first forces the LLM to output ONLY the name of the function, and then in a second phase forces the LLM to extract the parameters for that specific function. This greatly simplifies the decoding process and reduces hallucination.
- **Pre-decoded Vocabulary:** The Qwen tokenizer uses byte-level BPE. The entire vocabulary is pre-decoded to strings during initialization. Invalid UTF-8 sequences were filtered out, ensuring the matcher operates purely on strings.

## Performance analysis
- **Reliability:** By rejecting tokens that break the template string during the scan, the model is physically prevented from outputting invalid JSON. This results in a 100% success rate for JSON schema compliance.
- **Speed:** `np.argsort()` runs in highly optimized C code, and checking the top few tokens with native string slicing is computationally negligible compared to the LLM's forward pass. Processing a prompt takes a fraction of a second, comfortably passing the 5-minute requirement.

## Challenges faced
1. **The Trap of Complexity:** I initially implemented a massive system using an NFA and a Prefix Tree intersected via Depth-First Search. This was almost 700 lines of dense code. After realizing that this complexity was solving a problem that didn't exist: the overhead of checking a few strings per step is virtually zero, making the complex data structures completely useless.
2. **Vocabulary Decoding overhead:** Calling the tokenizer's decode function on 150k tokens individually during each generation step was too slow. This was solved by pre-decoding the vocabulary into a dictionary on startup.

## Testing strategy
The system was validated using the provided `data/input/function_calling_tests.json` file. Static analysis was enforced using `flake8` and `mypy --strict`.

## Example usage
```bash
uv run python -m src
```
This generates `data/output/function_calling_results.json` containing the mapped function calls.
