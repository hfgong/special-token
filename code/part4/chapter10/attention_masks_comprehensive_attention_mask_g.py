"""
Comprehensive attention mask generator for special tokens

Extracted from: part4/chapter10/attention_masks.tex
Block: 1
Lines: 90
"""

import torch
import torch.nn as nn
import numpy as np

class SpecialTokenMaskGenerator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.special_token_map = self._build_special_token_map()
        
    def _build_special_token_map(self):
        """Build mapping of special token types to their IDs."""
        token_map = {}
        
        # Standard special tokens
        for attr in ['cls_token_id', 'sep_token_id', 'pad_token_id', 
                     'mask_token_id', 'unk_token_id']:
            if hasattr(self.tokenizer, attr):
                token_id = getattr(self.tokenizer, attr)
                if token_id is not None:
                    token_map[attr.replace('_id', '')] = token_id
                    
        return token_map
        
    def create_attention_mask(self, input_ids, mask_type='bidirectional'):
        """Create sophisticated attention masks for special tokens."""
        batch_size, seq_len = input_ids.shape
        
        if mask_type == 'bidirectional':
            return self._create_bidirectional_mask(input_ids)
        elif mask_type == 'causal':
            return self._create_causal_mask(input_ids)
        elif mask_type == 'prefix_lm':
            return self._create_prefix_lm_mask(input_ids)
        elif mask_type == 'custom_special':
            return self._create_custom_special_mask(input_ids)
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
            
    def _create_bidirectional_mask(self, input_ids):
        """Standard bidirectional attention with padding mask."""
        # Basic padding mask
        padding_mask = (input_ids != self.special_token_map.get('pad_token', -1))
        
        # Expand to attention dimensions
        attention_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.expand(-1, 1, input_ids.size(1), -1)
        
        return attention_mask.float()
        
    def _create_causal_mask(self, input_ids):
        """Causal mask with special token considerations."""
        batch_size, seq_len = input_ids.shape
        
        # Create basic causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        
        # Special tokens can attend to all previous positions
        cls_token_id = self.special_token_map.get('cls_token')
        if cls_token_id is not None:
            cls_positions = (input_ids == cls_token_id)
            for batch_idx in range(batch_size):
                cls_pos = cls_positions[batch_idx].nonzero(as_tuple=True)[0]
                if len(cls_pos) > 0:
                    # CLS can attend to entire sequence
                    causal_mask[cls_pos[0], :] = 1
                    
        # Apply padding mask
        padding_mask = (input_ids != self.special_token_map.get('pad_token', -1))
        combined_mask = causal_mask.unsqueeze(0) * padding_mask.unsqueeze(1)
        
        return combined_mask.unsqueeze(1).float()
        
    def _create_prefix_lm_mask(self, input_ids):
        """Prefix LM mask where prefix tokens attend bidirectionally."""
        batch_size, seq_len = input_ids.shape
        
        # Find separator token positions
        sep_token_id = self.special_token_map.get('sep_token')
        
        masks = []
        for batch_idx in range(batch_size):
            mask = torch.zeros(seq_len, seq_len)
            
            if sep_token_id is not None:
                sep_positions = (input_ids[batch_idx] == sep_token_id).nonzero(as_tuple=True)[0]
                
                if len(sep_positions) > 0:
                    # Bidirectional attention for prefix (up to first SEP)
                    prefix_end = sep_positions[0].item()
                    mask[:prefix_end+1, :prefix_end+1] = 1
                    
                    # Causal attention for suffix (after SEP)
                    if prefix_end + 1 < seq_len:
                        causal_suffix = torch.tril(torch.ones(seq_len - prefix_end - 1, 
                                                              seq_len - prefix_end - 1))
                        mask[prefix_end+1:, prefix_end+1:] = causal_suffix
                        
                        # Suffix can attend to prefix
                        mask[prefix_end+1:, :prefix_end+1] = 1
                else:
                    # No separator found, use bidirectional
                    mask = torch.ones(seq_len, seq_len)
            else:
                # No separator token defined, use bidirectional
                mask = torch.ones(seq_len, seq_len)
                
            # Apply padding mask
            valid_positions = (input_ids[batch_idx] != self.special_token_map.get('pad_token', -1))
            mask = mask * valid_positions.unsqueeze(0) * valid_positions.unsqueeze(1)
            
            masks.append(mask)
            
        return torch.stack(masks).unsqueeze(1).float()