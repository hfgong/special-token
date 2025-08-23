"""
Learned position embeddings with special token support

Extracted from: part4/chapter10/position_encoding.tex
Block: 3
Lines: 138
"""

class LearnedPositionEmbedding(nn.Module):
    def __init__(self, max_length=512, d_model=768, special_token_ids=None):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model
        self.special_token_ids = set(special_token_ids or [])
        
        # Standard position embeddings
        self.position_embeddings = nn.Embedding(max_length, d_model)
        
        # Virtual positions for special tokens
        self.virtual_positions = nn.ParameterDict()
        self._initialize_virtual_positions()
        
        # Position adaptation networks
        self.position_adapters = nn.ModuleDict({
            'content_adapter': nn.Linear(d_model, d_model),
            'special_adapter': nn.Linear(d_model, d_model),
            'boundary_adapter': nn.Linear(d_model, d_model)
        })
        
    def _initialize_virtual_positions(self):
        """Initialize virtual positions for special tokens."""
        # Create virtual position embeddings that don't correspond to sequence positions
        virtual_positions = {
            'global_context': nn.Parameter(torch.randn(self.d_model) * 0.02),
            'sequence_start': nn.Parameter(torch.randn(self.d_model) * 0.02),
            'sequence_end': nn.Parameter(torch.randn(self.d_model) * 0.02),
            'segment_boundary': nn.Parameter(torch.randn(self.d_model) * 0.02),
            'meta_information': nn.Parameter(torch.randn(self.d_model) * 0.02)
        }
        
        for name, param in virtual_positions.items():
            self.virtual_positions[name] = param
            
    def forward(self, input_ids, position_ids=None):
        """Forward pass with special position handling."""
        batch_size, seq_len = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).expand(batch_size, -1)
            
        # Get base position embeddings
        base_positions = self.position_embeddings(position_ids)
        
        # Apply special token positioning
        enhanced_positions = self._apply_special_positioning(
            input_ids, position_ids, base_positions
        )
        
        return enhanced_positions
        
    def _apply_special_positioning(self, input_ids, position_ids, base_positions):
        """Apply special positioning for special tokens."""
        batch_size, seq_len, d_model = base_positions.shape
        enhanced_positions = base_positions.clone()
        
        for batch_idx in range(batch_size):
            sequence = input_ids[batch_idx]
            
            for pos_idx in range(seq_len):
                token_id = sequence[pos_idx].item()
                
                if token_id in self.special_token_ids:
                    # Determine virtual position type
                    virtual_type = self._get_virtual_position_type(
                        token_id, pos_idx, seq_len, sequence
                    )
                    
                    if virtual_type in self.virtual_positions:
                        # Replace with virtual position
                        virtual_pos = self.virtual_positions[virtual_type]
                        
                        # Adapt virtual position based on context
                        adapter = self._get_position_adapter(virtual_type)
                        adapted_pos = adapter(virtual_pos.unsqueeze(0)).squeeze(0)
                        
                        enhanced_positions[batch_idx, pos_idx] = adapted_pos
                        
        return enhanced_positions
        
    def _get_virtual_position_type(self, token_id, position, seq_len, sequence):
        """Determine the virtual position type for a special token."""
        if self._is_cls_token(token_id):
            return 'global_context'
        elif self._is_sep_token(token_id):
            if position < seq_len // 2:
                return 'segment_boundary'
            else:
                return 'sequence_end'
        elif position == 0:
            return 'sequence_start'
        elif position == seq_len - 1:
            return 'sequence_end'
        else:
            return 'meta_information'
            
    def _get_position_adapter(self, virtual_type):
        """Get the appropriate adapter for virtual position type."""
        if virtual_type in ['global_context', 'meta_information']:
            return self.position_adapters['special_adapter']
        elif virtual_type in ['segment_boundary', 'sequence_start', 'sequence_end']:
            return self.position_adapters['boundary_adapter']
        else:
            return self.position_adapters['content_adapter']

class ContextualPositionEncoding(nn.Module):
    def __init__(self, d_model=768, max_length=512):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        
        # Context-dependent position encoding
        self.context_projector = nn.Linear(d_model, d_model)
        self.position_generator = nn.Linear(d_model * 2, d_model)
        
        # Base position embeddings
        self.base_positions = nn.Embedding(max_length, d_model)
        
    def forward(self, token_embeddings, input_ids, position_ids=None):
        """Generate context-dependent position encodings."""
        batch_size, seq_len, d_model = token_embeddings.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).expand(batch_size, -1)
            
        # Get base positions
        base_pos = self.base_positions(position_ids)
        
        # Project token embeddings to position space
        context_features = self.context_projector(token_embeddings)
        
        # Combine context with base positions
        combined_features = torch.cat([context_features, base_pos], dim=-1)
        
        # Generate contextual positions
        contextual_positions = self.position_generator(combined_features)
        
        # Apply special token modifications
        modified_positions = self._modify_special_positions(
            contextual_positions, input_ids, token_embeddings
        )
        
        return modified_positions
        
    def _modify_special_positions(self, positions, input_ids, token_embeddings):
        """Modify positions for special tokens based on their semantic role."""
        batch_size, seq_len, d_model = positions.shape
        modified_positions = positions.clone()
        
        # Find special tokens and modify their positions
        for batch_idx in range(batch_size):
            sequence = input_ids[batch_idx]
            
            # CLS tokens get global context-aware positions
            cls_mask = self._create_cls_mask(sequence)
            if cls_mask.any():
                # Aggregate information from entire sequence
                sequence_context = token_embeddings[batch_idx].mean(dim=0, keepdim=True)
                global_position = self.context_projector(sequence_context)
                modified_positions[batch_idx, cls_mask] = global_position
                
            # SEP tokens get boundary-aware positions
            sep_mask = self._create_sep_mask(sequence)
            if sep_mask.any():
                # Use local context around separator
                for sep_idx in sep_mask.nonzero(as_tuple=True)[0]:
                    start_idx = max(0, sep_idx - 2)
                    end_idx = min(seq_len, sep_idx + 3)
                    local_context = token_embeddings[batch_idx, start_idx:end_idx].mean(dim=0)
                    boundary_position = self.context_projector(local_context)
                    modified_positions[batch_idx, sep_idx] = boundary_position
                    
        return modified_positions