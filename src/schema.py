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
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [FunctionDefinition.model_validate(item) for item in data]


def load_prompts(path: str) -> List[PromptTest]:
    """Carica e valida i prompt di test dal file JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [PromptTest.model_validate(item) for item in data]
