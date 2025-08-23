"""
Multi-scale position encoding for hierarchical processing

Extracted from: part4/chapter10/position_encoding.tex
Block: 4
Lines: 89
"""

class MultiScalePositionEncoding(nn.Module):
    def __init__(self, d_model=768, scales=[1, 4, 16, 64]):
        super().__init__()
        self.d_model = d_model
        self.scales = scales
        self.num_scales = len(scales)
        
        # Position encodings at different scales
        self.scale_encodings = nn.ModuleList([
            self._create_scale_encoding(scale) for scale in scales
        ])
        
        # Scale combination weights
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        
        # Special token scale preferences
        self.special_scale_preferences = nn.ParameterDict({
            'cls_scales': nn.Parameter(torch.softmax(torch.randn(self.num_scales), dim=0)),
            'sep_scales': nn.Parameter(torch.softmax(torch.randn(self.num_scales), dim=0)),
            'mask_scales': nn.Parameter(torch.softmax(torch.randn(self.num_scales), dim=0))
        })
        
    def _create_scale_encoding(self, scale):
        """Create position encoding for a specific scale."""
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
    def forward(self, input_ids, base_positions):
        """Generate multi-scale position encodings."""
        batch_size, seq_len, d_model = base_positions.shape
        
        # Compute position encodings at each scale
        scale_encodings = []
        for scale_idx, scale in enumerate(self.scales):
            # Downsample positions for this scale
            downsampled_positions = self._downsample_positions(base_positions, scale)
            
            # Apply scale-specific encoding
            scale_encoding = self.scale_encodings[scale_idx](downsampled_positions)
            
            # Upsample back to original resolution
            upsampled_encoding = self._upsample_positions(scale_encoding, scale, seq_len)
            scale_encodings.append(upsampled_encoding)
            
        # Combine scales with learned weights
        combined_encoding = self._combine_scales(scale_encodings, input_ids)
        
        return combined_encoding
        
    def _downsample_positions(self, positions, scale):
        """Downsample position encodings by averaging."""
        batch_size, seq_len, d_model = positions.shape
        
        if scale == 1:
            return positions
            
        # Reshape for downsampling
        pad_len = (scale - seq_len % scale) % scale
        if pad_len > 0:
            padding = torch.zeros(batch_size, pad_len, d_model, device=positions.device)
            padded_positions = torch.cat([positions, padding], dim=1)
        else:
            padded_positions = positions
            
        # Average pool with scale as kernel size
        downsampled = padded_positions.view(
            batch_size, -1, scale, d_model
        ).mean(dim=2)
        
        return downsampled
        
    def _upsample_positions(self, scale_encoding, scale, target_length):
        """Upsample position encodings to target length."""
        if scale == 1:
            return scale_encoding[:, :target_length]
            
        # Repeat each encoding 'scale' times
        batch_size, downsampled_len, d_model = scale_encoding.shape
        upsampled = scale_encoding.unsqueeze(2).expand(-1, -1, scale, -1)
        upsampled = upsampled.contiguous().view(batch_size, -1, d_model)
        
        return upsampled[:, :target_length]
        
    def _combine_scales(self, scale_encodings, input_ids):
        """Combine multi-scale encodings with token-specific preferences."""
        batch_size, seq_len = input_ids.shape
        
        # Stack scale encodings
        stacked_encodings = torch.stack(scale_encodings, dim=-1)  # [B, L, D, S]
        
        # Default combination weights
        default_weights = self.scale_weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        combined_weights = default_weights.expand(batch_size, seq_len, 1, -1)
        
        # Apply special token preferences
        for batch_idx in range(batch_size):
            sequence = input_ids[batch_idx]
            
            for pos_idx in range(seq_len):
                token_id = sequence[pos_idx].item()
                
                if self._is_cls_token(token_id):
                    combined_weights[batch_idx, pos_idx, 0] = self.special_scale_preferences['cls_scales']
                elif self._is_sep_token(token_id):
                    combined_weights[batch_idx, pos_idx, 0] = self.special_scale_preferences['sep_scales']
                elif self._is_mask_token(token_id):
                    combined_weights[batch_idx, pos_idx, 0] = self.special_scale_preferences['mask_scales']
                    
        # Weighted combination
        combined_encoding = (stacked_encodings * combined_weights).sum(dim=-1)
        
        return combined_encoding