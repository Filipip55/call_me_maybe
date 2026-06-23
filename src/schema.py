import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class ParameterDefinition(BaseModel):
    type: str
    description: Optional[str] = None


class FunctionReturns(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterDefinition]
    returns: FunctionReturns


class PromptTest(BaseModel):
    prompt: str


class FunctionCallingResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]


def load_functions_definition(path: str) -> List[FunctionDefinition]:
    with open(path, 'r') as f:
        data = json.load(f)
    return [FunctionDefinition.model_validate(item) for item in data]


def load_prompts(path: str) -> List[PromptTest]:
    with open(path, 'r') as f:
        data = json.load(f)
    return [PromptTest.model_validate(item) for item in data]
