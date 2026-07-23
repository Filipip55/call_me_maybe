import json
from typing import Dict, List, Any
from .fsm import State, advance_nfa, epsilon_closure


class TrieNode:
    """Nodo del Prefix Tree per il vocabolario."""
    def __init__(self) -> None:
        self.children: Dict[str, 'TrieNode'] = {}
        self.token_id: int | None = None


class Vocabulary:
    """Struttura Trie che memorizza i token validi dell'LLM decodificati in stringhe."""
    def __init__(self, llm: Any) -> None:
        vocab_path = llm.get_path_to_vocab_file()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        self.root = TrieNode()

        # Costruisce il Trie decodificando i token in stringhe
        for token_str, token_id in vocab.items():
            decoded_str = llm.decode([token_id])
            if '\ufffd' in decoded_str:
                # Salta le sequenze UTF-8 non valide
                continue
            if not decoded_str:
                continue
            self._insert(decoded_str, token_id)

    def _insert(self, decoded_str: str, token_id: int) -> None:
        """Inserisce una stringa e il relativo token ID all'interno del Trie."""
        node = self.root
        for char in decoded_str:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.token_id = token_id


class JSONStateMachine:
    """Gestisce l'evoluzione dello stato e il calcolo dei token permessi, tramite intersezione NFA/Trie."""
    def __init__(self, start_state: State, end_state: State):
        self.start_state = start_state
        self.end_state = end_state
        self.current_states = epsilon_closure({self.start_state})
        self.memo: Dict[Any, frozenset] = {}
        self.valid_tokens_memo: Dict[frozenset, List[int]] = {}

    def advance(self, text: str) -> bool:
        """Avanza la macchina a stati consumando i caratteri della stringa in input. Restituisce True se valido."""
        states = self.current_states
        for char in text:
            states = advance_nfa(states, char)
            if not states:
                return False
        self.current_states = states
        return True

    def get_valid_next_tokens(self, vocab: Vocabulary) -> List[int]:
        """Restituisce una lista di ID di token che rappresentano continuazioni valide dallo stato attuale."""
        states_key = frozenset(self.current_states)
        if states_key in self.valid_tokens_memo:
            return self.valid_tokens_memo[states_key]

        valid_tokens = []

        def dfs(node: TrieNode, states: frozenset) -> None:
            if node.token_id is not None:
                valid_tokens.append(node.token_id)

            for char, child_node in node.children.items():
                key = (states, char)
                if key not in self.memo:
                    self.memo[key] = frozenset(advance_nfa(set(states), char))
                next_states = self.memo[key]

                if next_states:
                    dfs(child_node, next_states)

        dfs(vocab.root, states_key)
        result = list(set(valid_tokens))
        self.valid_tokens_memo[states_key] = result
        return result

    def is_finished(self) -> bool:
        """Controlla se lo stato finale è stato raggiunto."""
        return self.end_state in self.current_states
