"""
Flexible position encoding for special tokens

Extracted from: part4/chapter10/position_encoding.tex
Block: 1
Lines: 86
"""

import torch
import torch.nn as nn
import math

class SpecialTokenPositionEncoder:
    def __init__(self, max_length=512, d_model=768, special_token_map=None):
        self.max_length = max_length
        self.d_model = d_model
        self.special_token_map = special_token_map or {}
        
        # Standard sinusoidal position encodings
        self.pe_matrix = self._create_sinusoidal_encodings()
        
        # Learnable special position encodings
        self.special_position_embeddings = nn.ParameterDict()
        self._initialize_special_positions()
        
    def _create_sinusoidal_encodings(self):
        """Create standard sinusoidal position encodings."""
        pe = torch.zeros(self.max_length, self.d_model)
        position = torch.arange(0, self.max_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() *
                           -(math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
        
    def _initialize_special_positions(self):
        """Initialize learnable position encodings for special tokens."""
        special_positions = {
            'cls_position': nn.Parameter(torch.randn(self.d_model) * 0.02),
            'sep_position': nn.Parameter(torch.randn(self.d_model) * 0.02),
            'mask_position': nn.Parameter(torch.randn(self.d_model) * 0.02),
            'global_position': nn.Parameter(torch.randn(self.d_model) * 0.02),
            'boundary_position': nn.Parameter(torch.randn(self.d_model) * 0.02)
        }
        
        for name, param in special_positions.items():
            self.special_position_embeddings[name] = param
            
    def encode_positions(self, input_ids, position_strategy='adaptive'):
        """Encode positions for input sequence with special token handling."""
        batch_size, seq_len = input_ids.shape
        
        if position_strategy == 'adaptive':
            return self._adaptive_position_encoding(input_ids)
        elif position_strategy == 'fixed_special':
            return self._fixed_special_encoding(input_ids)
        elif position_strategy == 'relative':
            return self._relative_position_encoding(input_ids)
        elif position_strategy == 'learned':
            return self._learned_position_encoding(input_ids)
        else:
            return self._standard_encoding(input_ids)
            
    def _adaptive_position_encoding(self, input_ids):
        """Adaptive position encoding that adjusts for special tokens."""
        batch_size, seq_len = input_ids.shape
        position_encodings = torch.zeros(batch_size, seq_len, self.d_model)
        
        for batch_idx in range(batch_size):
            sequence = input_ids[batch_idx]
            positions = self._compute_adaptive_positions(sequence)
            
            for pos_idx, position_type in enumerate(positions):
                if position_type == 'standard':
                    # Use regular sinusoidal encoding
                    actual_pos = self._get_content_position(sequence, pos_idx)
                    position_encodings[batch_idx, pos_idx] = self.pe_matrix[actual_pos]
                elif position_type in self.special_position_embeddings:
                    # Use special position encoding
                    position_encodings[batch_idx, pos_idx] = self.special_position_embeddings[position_type]
                    
        return position_encodings
        
    def _compute_adaptive_positions(self, sequence):
        """Compute position types for each token in sequence."""
        positions = []
        content_position = 0
        
        for token_id in sequence:
            if self._is_cls_token(token_id):
                positions.append('cls_position')
            elif self._is_sep_token(token_id):
                positions.append('sep_position')
            elif self._is_mask_token(token_id):
                positions.append('mask_position')
            elif self._is_special_token(token_id):
                positions.append('global_position')
            else:
                positions.append('standard')
                content_position += 1
                
        return positions
        
    def _get_content_position(self, sequence, current_idx):
        """Get the content position for regular tokens."""
        content_pos = 0
        for i in range(current_idx):
            if not self._is_special_token(sequence[i]):
                content_pos += 1
        return min(content_pos, self.max_length - 1)