import re
import sys
from pathlib import Path
from typing import List, Dict

sys.path.append(str(Path(__file__).parent.parent))
from config import PAD_ID, SOS_ID, EOS_ID


class LaTeXTokenizer:
    def __init__(self, latex_strings: List[str] = None):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Special tokens with fixed IDs from config; UNK is placed after them.
        self.token_to_id['<PAD>'] = PAD_ID
        self.token_to_id['<SOS>'] = SOS_ID
        self.token_to_id['<EOS>'] = EOS_ID

        self.id_to_token[PAD_ID] = '<PAD>'
        self.id_to_token[SOS_ID] = '<SOS>'
        self.id_to_token[EOS_ID] = '<EOS>'

        # Unknown token: any unseen symbol will map to this ID.
        self.UNK_ID = max(PAD_ID, SOS_ID, EOS_ID) + 1
        self.token_to_id['<UNK>'] = self.UNK_ID
        self.id_to_token[self.UNK_ID] = '<UNK>'
        
        if latex_strings:
            self._build_vocab(latex_strings)
    
    def _tokenize(self, latex_string: str) -> List[str]:
        tokens = []
        i = 0
        while i < len(latex_string):
            if latex_string[i] == '\\':
                j = i + 1
                while j < len(latex_string) and latex_string[j].isalpha():
                    j += 1
                tokens.append(latex_string[i:j])
                i = j
            elif latex_string[i] in ['{', '}', '[', ']', '(', ')', '^', '_']:
                tokens.append(latex_string[i])
                i += 1
            elif latex_string[i].isspace():
                i += 1
            else:
                tokens.append(latex_string[i])
                i += 1
        return tokens
    
    def _build_vocab(self, latex_strings: List[str]):
        vocab_set = set()
        
        for latex_str in latex_strings:
            tokens = self._tokenize(latex_str)
            vocab_set.update(tokens)
        
        # Start assigning new IDs after all special tokens (including UNK).
        next_id = max(self.token_to_id.values()) + 1
        for token in sorted(vocab_set):
            if token not in self.token_to_id:
                self.token_to_id[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1
    
    def encode(self, latex_string: str) -> List[int]:
        tokens = self._tokenize(latex_string)
        ids = [SOS_ID]
        for token in tokens:
            # Map unknown tokens to <UNK> instead of failing.
            token_id = self.token_to_id.get(token, self.UNK_ID)
            ids.append(token_id)
        ids.append(EOS_ID)
        return ids
    
    def decode(self, token_ids: List[int]) -> str:
        tokens = []
        for token_id in token_ids:
            if token_id == PAD_ID:
                continue
            if token_id == SOS_ID:
                continue
            if token_id == EOS_ID:
                break
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                raise ValueError(f"Unknown token ID: {token_id}")
        return ''.join(tokens)
    
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)


if __name__ == "__main__":
    test_formulas = [
        r"\frac{a}{b} = c",
        r"\sum_{i=1}^{n} x_i",
        r"\int_{0}^{\infty} e^{-x} dx"
    ]
    
    tokenizer = LaTeXTokenizer(test_formulas)
    
    test_formula = r"\frac{a}{b} = c"
    print(f"Original: {test_formula}")
    
    encoded = tokenizer.encode(test_formula)
    print(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    
    print(f"\nVocab size: {tokenizer.vocab_size}")
    print(f"Vocabulary (first 20): {sorted(tokenizer.token_to_id.items(), key=lambda x: x[1])[:20]}")

