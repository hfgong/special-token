"""
Special token vocabulary management

Extracted from: part1/chapter01/tokenization_and_insertion.tex
Block: 7
Lines: 51
"""

class SpecialTokenVocabularyManager:
    def __init__(self, base_vocab_size=30000):
        self.base_vocab_size = base_vocab_size
        self.special_tokens = {}
        self.reserved_ids = set()
        
    def reserve_special_token_space(self, num_special_tokens=100):
        """Reserve space at the end of vocabulary for special tokens"""
        start_id = self.base_vocab_size
        end_id = start_id + num_special_tokens
        self.reserved_ids = set(range(start_id, end_id))
        return start_id, end_id
    
    def add_special_token(self, token_str, token_id=None):
        """Add a special token to the vocabulary"""
        if token_id is None:
            # Find next available ID in reserved space
            available_ids = self.reserved_ids - set(self.special_tokens.values())
            if not available_ids:
                raise ValueError("No available special token IDs")
            token_id = min(available_ids)
        
        if token_id not in self.reserved_ids:
            raise ValueError(f"Token ID {token_id} not in reserved space")
        
        self.special_tokens[token_str] = token_id
        return token_id
    
    def batch_add_special_tokens(self, token_list):
        """Add multiple special tokens efficiently"""
        available_ids = sorted(self.reserved_ids - set(self.special_tokens.values()))
        
        if len(token_list) > len(available_ids):
            raise ValueError("Not enough reserved space for all tokens")
        
        for i, token_str in enumerate(token_list):
            self.special_tokens[token_str] = available_ids[i]
        
        return {token: available_ids[i] for i, token in enumerate(token_list)}
    
    def export_vocabulary_config(self):
        """Export special token configuration for model serialization"""
        return {
            'base_vocab_size': self.base_vocab_size,
            'special_tokens': self.special_tokens,
            'reserved_space': list(self.reserved_ids)
        }
    
    def validate_token_consistency(self, other_vocab_config):
        """Validate consistency with another vocabulary configuration"""
        conflicts = []
        
        for token, token_id in self.special_tokens.items():
            if token in other_vocab_config['special_tokens']:
                other_id = other_vocab_config['special_tokens'][token]
                if token_id != other_id:
                    conflicts.append({
                        'token': token,
                        'current_id': token_id,
                        'other_id': other_id
                    })
        
        return conflicts