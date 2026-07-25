import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class ParameterDefinition(BaseModel):
    """Definisce un singolo parametro di una funzione (tipo e descrizione)."""
    type: str
    description: Optional[str] = None


class FunctionReturns(BaseModel):
    """Definisce il tipo di ritorno di una funzione."""
    type: str


class FunctionDefinition(BaseModel):
    """Schema di una funzione disponibile per l'LLM."""
    name: str
    description: str
    parameters: Dict[str, ParameterDefinition]
    returns: FunctionReturns


class PromptTest(BaseModel):
    """Rappresenta un singolo prompt di test inviato dall'utente."""
    prompt: str


class FunctionCallingResult(BaseModel):
    """Struttura del risultato finale generato dall'LLM."""
    prompt: str
    name: str
    parameters: Dict[str, Any]


def load_functions_definition(path: str) -> List[FunctionDefinition]:
    """Carica e valida le definizioni delle funzioni dal file JSON."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"Error: Function definitions file '{path}' must contain a JSON list.")
            return []
        functions = []
        for item in data:
            try:
                functions.append(FunctionDefinition.model_validate(item))
            except Exception as e:
                print(f"Warning: Invalid function schema item skipped: {e}")
        return functions
    except FileNotFoundError:
        print(f"Error: Function definitions file not found: '{path}'")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Malformed JSON in function definitions file '{path}': {e}")
        return []
    except Exception as e:
        print(f"Error loading function definitions from '{path}': {e}")
        return []


def load_prompts(path: str) -> List[PromptTest]:
    """Carica e valida i prompt di test dal file JSON."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"Error: Prompts file '{path}' must contain a JSON list.")
            return []
        prompts = []
        for item in data:
            try:
                if isinstance(item, str):
                    prompts.append(PromptTest(prompt=item))
                else:
                    prompts.append(PromptTest.model_validate(item))
            except Exception as e:
                print(f"Warning: Invalid prompt item skipped: {e}")
        return prompts
    except FileNotFoundError:
        print(f"Error: Prompts input file not found: '{path}'")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Malformed JSON in prompts input file '{path}': {e}")
        return []
    except Exception as e:
        print(f"Error loading prompts from '{path}': {e}")
        return []

