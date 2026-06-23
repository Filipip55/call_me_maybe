import json
from typing import Dict, List, Set, Any
from .fsm import State, advance_nfa, epsilon_closure


class TrieNode:
    def __init__(self) -> None:
        self.children: Dict[str, 'TrieNode'] = {}
        self.token_id: int | None = None


class Vocabulary:
    def __init__(self, llm: Any) -> None:
        vocab_path = llm.get_path_to_vocab_file()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        self.root = TrieNode()

        # Build Trie using actual decoded strings
        for token_str, token_id in vocab.items():
            decoded_str = llm.decode([token_id])
            if '\ufffd' in decoded_str:
                # Skip invalid utf-8 sequences
                continue
            if not decoded_str:
                continue
            self._insert(decoded_str, token_id)

    def _insert(self, decoded_str: str, token_id: int) -> None:
        node = self.root
        for char in decoded_str:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.token_id = token_id


class JSONStateMachine:
    def __init__(self, start_state: State, end_state: State):
        self.start_state = start_state
        self.end_state = end_state
        self.current_states = epsilon_closure({self.start_state})
        self.memo: Dict[Any, frozenset] = {}

    def advance(self, text: str) -> bool:
        """Advances the state machine by the given text. Returns True if successful."""
        states = self.current_states
        for char in text:
            states = advance_nfa(states, char)
            if not states:
                return False
        self.current_states = states
        return True

    def get_valid_next_tokens(self, vocab: Vocabulary) -> List[int]:
        """Returns a list of token IDs that are valid continuations from the current state."""
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

        dfs(vocab.root, frozenset(self.current_states))
        return list(set(valid_tokens))

    def is_finished(self) -> bool:
        return self.end_state in self.current_states
