from typing import Dict, Set, List
from .schema import FunctionDefinition


class State:
    def __init__(self) -> None:
        self.transitions: Dict[str, Set['State']] = {}
        self.epsilon_transitions: Set['State'] = set()
        self.is_final = False

    def add_transition(self, char: str, state: 'State') -> None:
        if char not in self.transitions:
            self.transitions[char] = set()
        self.transitions[char].add(state)

    def add_epsilon(self, state: 'State') -> None:
        self.epsilon_transitions.add(state)


def epsilon_closure(states: Set[State]) -> Set[State]:
    closure = set(states)
    stack = list(states)
    while stack:
        state = stack.pop()
        for next_state in state.epsilon_transitions:
            if next_state not in closure:
                closure.add(next_state)
                stack.append(next_state)
    return closure


def advance_nfa(states: Set[State], char: str) -> Set[State]:
    next_states = set()
    for state in states:
        if char in state.transitions:
            next_states.update(state.transitions[char])
        if '*' in state.transitions:
            if char != '"' and char != '\\':
                next_states.update(state.transitions['*'])
    return epsilon_closure(next_states)


def build_literal(text: str) -> tuple[State, State]:
    start = State()
    current = start
    for char in text:
        next_state = State()
        current.add_transition(char, next_state)
        current = next_state
    return start, current


def build_whitespace() -> tuple[State, State]:
    start = State()
    end = State()
    start.add_epsilon(end)
    for ws in " \t\n\r":
        start.add_transition(ws, start)
    return start, end


def build_number() -> tuple[State, State]:
    start = State()
    end = State()

    minus = State()
    start.add_transition('-', minus)
    start.add_epsilon(minus)

    int_part = State()
    for d in "0123456789":
        minus.add_transition(d, int_part)
        int_part.add_transition(d, int_part)

    int_part.add_epsilon(end)

    dot = State()
    int_part.add_transition('.', dot)
    frac_part = State()
    for d in "0123456789":
        dot.add_transition(d, frac_part)
        frac_part.add_transition(d, frac_part)

    frac_part.add_epsilon(end)
    return start, end


def build_string() -> tuple[State, State]:
    start = State()
    end = State()

    q1 = State()
    start.add_transition('"', q1)

    content = State()
    q1.add_epsilon(content)

    content.add_transition('*', content)

    escape = State()
    content.add_transition('\\', escape)
    for ec in '"\\/bfnrt':
        escape.add_transition(ec, content)

    content.add_transition('"', end)
    return start, end


def build_boolean() -> tuple[State, State]:
    start = State()
    end = State()
    t_start, t_end = build_literal("true")
    f_start, f_end = build_literal("false")
    start.add_epsilon(t_start)
    start.add_epsilon(f_start)
    t_end.add_epsilon(end)
    f_end.add_epsilon(end)
    return start, end


def concat_nfas(nfas: List[tuple[State, State]]) -> tuple[State, State]:
    if not nfas:
        s = State()
        return s, s
    start, current_end = nfas[0]
    for n_start, n_end in nfas[1:]:
        current_end.add_epsilon(n_start)
        current_end = n_end
    return start, current_end


def build_choice(nfas: List[tuple[State, State]]) -> tuple[State, State]:
    start = State()
    end = State()
    for n_start, n_end in nfas:
        start.add_epsilon(n_start)
        n_end.add_epsilon(end)
    return start, end


def build_schema_nfa(functions: List[FunctionDefinition]) -> tuple[State, State]:
    nfas = []

    for fn in functions:
        parts = []
        parts.append(build_literal('{"name":'))
        parts.append(build_whitespace())
        parts.append(build_literal(f'"{fn.name}"'))
        parts.append(build_whitespace())
        parts.append(build_literal(',"parameters":'))
        parts.append(build_whitespace())
        parts.append(build_literal('{'))
        parts.append(build_whitespace())

        params = list(fn.parameters.items())
        for i, (p_name, p_def) in enumerate(params):
            parts.append(build_literal(f'"{p_name}":'))
            parts.append(build_whitespace())

            if p_def.type == "number":
                parts.append(build_number())
            elif p_def.type == "string":
                parts.append(build_string())
            elif p_def.type == "boolean":
                parts.append(build_boolean())
            else:
                # Default fallback
                parts.append(build_string())

            parts.append(build_whitespace())
            if i < len(params) - 1:
                parts.append(build_literal(','))
                parts.append(build_whitespace())

        parts.append(build_literal('}'))
        parts.append(build_whitespace())
        parts.append(build_literal('}'))

        fn_nfa = concat_nfas(parts)
        nfas.append(fn_nfa)

    final_choice = build_choice(nfas)
    return final_choice
