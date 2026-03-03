import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        ids = []
        special_tokens = ["<PAD>","<UNK>", "<BOS>", "<EOS>"]
        self.word_to_id = {word:id for id,word in enumerate(special_tokens)}
        self.id_to_word = {id:word for id,word in enumerate(special_tokens)}
        
        self.vocab_size = len(self.word_to_id)
        for text in texts:
            for t in text.split():
                if t not in self.word_to_id:
                    self.word_to_id[t] = self.vocab_size
                    self.id_to_word[self.vocab_size] = t
                    self.vocab_size += 1
                
        
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        return [
                    self.word_to_id.get(t, self.word_to_id[self.unk_token])
                    for t in text.split()
                ]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        return ' '.join([self.id_to_word[i] for i in ids])
        
