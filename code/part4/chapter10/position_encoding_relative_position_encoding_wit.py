"""
Relative position encoding with special token awareness

Extracted from: part4/chapter10/position_encoding.tex
Block: 2
Lines: 81
"""

class RelativePositionEncoding(nn.Module):
    def __init__(self, d_model=768, max_relative_distance=128):
        super().__init__()
        self.d_model = d_model
        self.max_relative_distance = max_relative_distance
        
        # Relative position embeddings
        self.relative_position_embeddings = nn.Embedding(
            2 * max_relative_distance + 1, d_model
        )
        
        # Special token relation embeddings
        self.special_relations = nn.ParameterDict({
            'cls_to_content': nn.Parameter(torch.randn(d_model) * 0.02),
            'content_to_cls': nn.Parameter(torch.randn(d_model) * 0.02),
            'sep_to_content': nn.Parameter(torch.randn(d_model) * 0.02),
            'content_to_sep': nn.Parameter(torch.randn(d_model) * 0.02),
            'special_to_special': nn.Parameter(torch.randn(d_model) * 0.02),
            'mask_to_content': nn.Parameter(torch.randn(d_model) * 0.02),
            'content_to_mask': nn.Parameter(torch.randn(d_model) * 0.02)
        })
        
    def forward(self, input_ids, query_pos, key_pos):
        """Compute relative position encodings."""
        batch_size, seq_len = input_ids.shape
        
        # Compute standard relative distances
        relative_distances = query_pos.unsqueeze(-1) - key_pos.unsqueeze(-2)
        
        # Clamp distances
        clamped_distances = torch.clamp(
            relative_distances,
            -self.max_relative_distance,
            self.max_relative_distance
        )
        
        # Convert to embedding indices
        embedding_indices = clamped_distances + self.max_relative_distance
        
        # Get base relative embeddings
        relative_embeddings = self.relative_position_embeddings(embedding_indices)
        
        # Apply special token modifications
        special_embeddings = self._apply_special_relations(
            input_ids, query_pos, key_pos, relative_embeddings
        )
        
        return special_embeddings
        
    def _apply_special_relations(self, input_ids, query_pos, key_pos, base_embeddings):
        """Apply special token relation modifications."""
        batch_size, seq_len_q, seq_len_k, d_model = base_embeddings.shape
        
        for batch_idx in range(batch_size):
            sequence = input_ids[batch_idx]
            
            for q_idx in range(seq_len_q):
                for k_idx in range(seq_len_k):
                    query_token = sequence[query_pos[batch_idx, q_idx]]
                    key_token = sequence[key_pos[batch_idx, k_idx]]
                    
                    # Determine relation type
                    relation_type = self._get_relation_type(query_token, key_token)
                    
                    if relation_type in self.special_relations:
                        # Modify embedding based on special relation
                        special_embedding = self.special_relations[relation_type]
                        base_embeddings[batch_idx, q_idx, k_idx] += special_embedding
                        
        return base_embeddings
        
    def _get_relation_type(self, query_token, key_token):
        """Determine the type of relation between two tokens."""
        query_is_cls = self._is_cls_token(query_token)
        key_is_cls = self._is_cls_token(key_token)
        query_is_sep = self._is_sep_token(query_token)
        key_is_sep = self._is_sep_token(key_token)
        query_is_mask = self._is_mask_token(query_token)
        key_is_mask = self._is_mask_token(key_token)
        
        query_is_special = query_is_cls or query_is_sep or query_is_mask
        key_is_special = key_is_cls or key_is_sep or key_is_mask
        
        if query_is_cls and not key_is_special:
            return 'cls_to_content'
        elif not query_is_special and key_is_cls:
            return 'content_to_cls'
        elif query_is_sep and not key_is_special:
            return 'sep_to_content'
        elif not query_is_special and key_is_sep:
            return 'content_to_sep'
        elif query_is_mask and not key_is_special:
            return 'mask_to_content'
        elif not query_is_special and key_is_mask:
            return 'content_to_mask'
        elif query_is_special and key_is_special:
            return 'special_to_special'
        else:
            return None  # Use base embedding