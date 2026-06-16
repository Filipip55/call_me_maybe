from llm_sdk import Small_LLM_Model

# IMPORTANT RULES:
# main model is qwen 3-0.6b
# classes use pydantic for validation
# numpy and json packages allowed
# no private methods or attributes from llm_sdk
# use uv for virtual env and dependency management
# look up what tokens and logits are DONE
# USAGE:

def main() -> None:
	llm = Small_LLM_Model()
	input_ids = llm.encode("Ciao ti sto testando")
	print(llm.decode(input_ids))

if __name__ == "__main__":
	main()
