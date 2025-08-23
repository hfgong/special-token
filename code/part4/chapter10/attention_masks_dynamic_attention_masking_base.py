"""
Dynamic attention masking based on content

Extracted from: part4/chapter10/attention_masks.tex
Block: 3
Lines: 119
"""

class DynamicAttentionMasking(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Learned masking parameters
        self.mask_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Special token attention controllers
        self.special_token_controllers = nn.ModuleDict({
            'cls_controller': nn.Linear(hidden_size, num_heads),
            'sep_controller': nn.Linear(hidden_size, num_heads),
            'mask_controller': nn.Linear(hidden_size, num_heads)
        })
        
    def forward(self, hidden_states, input_ids, base_attention_mask):
        """Generate dynamic attention masks."""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Predict attention weights for each position
        attention_weights = self.mask_predictor(hidden_states).squeeze(-1)
        
        # Create position-wise mask
        position_mask = attention_weights.unsqueeze(1) * attention_weights.unsqueeze(2)
        
        # Apply special token rules
        special_token_mask = self._apply_special_token_rules(
            hidden_states, input_ids, position_mask
        )
        
        # Combine with base mask
        final_mask = base_attention_mask * special_token_mask
        
        return final_mask
        
    def _apply_special_token_rules(self, hidden_states, input_ids, position_mask):
        """Apply learned rules for special token attention."""
        batch_size, seq_len, _ = hidden_states.shape
        special_mask = position_mask.clone()
        
        # Process each special token type
        special_tokens = {
            'cls_token_id': 'cls_controller',
            'sep_token_id': 'sep_controller', 
            'mask_token_id': 'mask_controller'
        }
        
        for token_attr, controller_name in special_tokens.items():
            token_id = getattr(self.tokenizer, token_attr, None)
            if token_id is not None and controller_name in self.special_token_controllers:
                controller = self.special_token_controllers[controller_name]
                
                # Find positions of this special token
                token_positions = (input_ids == token_id)
                
                if token_positions.any():
                    # Get hidden states for these positions
                    token_hidden = hidden_states[token_positions]
                    
                    # Predict attention modulation
                    attention_modulation = controller(token_hidden)  # [num_tokens, num_heads]
                    
                    # Apply modulation to attention mask
                    for batch_idx in range(batch_size):
                        batch_positions = token_positions[batch_idx].nonzero(as_tuple=True)[0]
                        
                        for i, pos in enumerate(batch_positions):
                            # Modulate attention from this position
                            modulation = attention_modulation[i].mean()  # Average over heads
                            special_mask[batch_idx, pos, :] *= modulation
                            
        return special_mask

class ConditionalMasking:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def create_task_conditional_mask(self, input_ids, task_type='classification'):
        """Create attention masks based on task requirements."""
        batch_size, seq_len = input_ids.shape
        
        if task_type == 'classification':
            return self._classification_mask(input_ids)
        elif task_type == 'generation':
            return self._generation_mask(input_ids)
        elif task_type == 'question_answering':
            return self._qa_mask(input_ids)
        elif task_type == 'summarization':
            return self._summarization_mask(input_ids)
        else:
            # Default bidirectional mask
            return self._default_mask(input_ids)
            
    def _classification_mask(self, input_ids):
        """Attention mask optimized for classification tasks."""
        batch_size, seq_len = input_ids.shape
        
        # Full bidirectional attention
        attention_mask = torch.ones(batch_size, seq_len, seq_len)
        
        # CLS token gets enhanced attention to all positions
        cls_token_id = getattr(self.tokenizer, 'cls_token_id', None)
        if cls_token_id is not None:
            cls_positions = (input_ids == cls_token_id)
            
            # Boost attention from CLS to all other tokens
            for batch_idx in range(batch_size):
                cls_pos = cls_positions[batch_idx].nonzero(as_tuple=True)[0]
                if len(cls_pos) > 0:
                    attention_mask[batch_idx, cls_pos[0], :] = 1.5  # Enhanced attention
                    
        # Apply padding mask
        return self._apply_padding_mask(attention_mask, input_ids)
        
    def _generation_mask(self, input_ids):
        """Causal mask for generation tasks."""
        seq_len = input_ids.size(1)
        
        # Causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        
        # Special tokens can attend to full context
        special_tokens = [
            getattr(self.tokenizer, 'cls_token_id', None),
            getattr(self.tokenizer, 'sep_token_id', None)
        ]
        
        for batch_idx in range(input_ids.size(0)):
            for token_id in special_tokens:
                if token_id is not None:
                    positions = (input_ids[batch_idx] == token_id).nonzero(as_tuple=True)[0]
                    for pos in positions:
                        causal_mask[pos, :pos+1] = 1  # Can attend to all previous
                        
        mask = causal_mask.unsqueeze(0).expand(input_ids.size(0), -1, -1)
        return self._apply_padding_mask(mask, input_ids)
        
    def _apply_padding_mask(self, attention_mask, input_ids):
        """Apply padding mask to attention matrix."""
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        if pad_token_id is not None:
            padding_mask = (input_ids != pad_token_id)
            attention_mask = attention_mask * padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)
            
        return attention_mask.unsqueeze(1).float()