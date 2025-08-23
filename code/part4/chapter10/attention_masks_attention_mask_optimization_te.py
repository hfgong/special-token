"""
Attention mask optimization techniques

Extracted from: part4/chapter10/attention_masks.tex
Block: 4
Lines: 89
"""

class AttentionMaskOptimizer:
    def __init__(self):
        self.mask_cache = {}
        self.optimization_stats = {}
        
    def optimize_mask_computation(self, input_ids, mask_type='bidirectional'):
        """Optimize mask computation with caching and vectorization."""
        
        # Create cache key
        cache_key = self._create_cache_key(input_ids, mask_type)
        
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]
            
        # Vectorized mask computation
        if mask_type == 'bidirectional':
            mask = self._vectorized_bidirectional_mask(input_ids)
        elif mask_type == 'causal':
            mask = self._vectorized_causal_mask(input_ids)
        else:
            mask = self._fallback_mask_computation(input_ids, mask_type)
            
        # Cache result
        if len(self.mask_cache) < 1000:  # Prevent unlimited growth
            self.mask_cache[cache_key] = mask
            
        return mask
        
    def _vectorized_bidirectional_mask(self, input_ids):
        """Highly optimized bidirectional mask computation."""
        batch_size, seq_len = input_ids.shape
        
        # Vectorized padding mask
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', -1)
        valid_mask = (input_ids != pad_token_id).float()
        
        # Outer product for attention mask
        attention_mask = torch.bmm(
            valid_mask.unsqueeze(2),
            valid_mask.unsqueeze(1)
        )
        
        return attention_mask.unsqueeze(1)
        
    def _vectorized_causal_mask(self, input_ids):
        """Optimized causal mask with special token handling."""
        batch_size, seq_len = input_ids.shape
        
        # Base causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        
        # Apply to batch
        batch_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Padding mask
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', -1)
        valid_mask = (input_ids != pad_token_id).float()
        
        # Combine masks
        final_mask = batch_mask * valid_mask.unsqueeze(1) * valid_mask.unsqueeze(2)
        
        return final_mask.unsqueeze(1)
        
    def compress_sparse_mask(self, attention_mask, sparsity_threshold=0.1):
        """Compress sparse attention masks for memory efficiency."""
        
        # Identify sparse regions
        density = attention_mask.mean(dim=-1, keepdim=True)
        sparse_regions = density < sparsity_threshold
        
        # Create compressed representation
        compressed_mask = attention_mask.clone()
        compressed_mask[sparse_regions.expand_as(attention_mask)] = 0
        
        # Store compression statistics
        original_nonzeros = attention_mask.nonzero().size(0)
        compressed_nonzeros = compressed_mask.nonzero().size(0)
        compression_ratio = compressed_nonzeros / original_nonzeros
        
        self.optimization_stats['compression_ratio'] = compression_ratio
        
        return compressed_mask
        
    def adaptive_masking_threshold(self, attention_weights, percentile=90):
        """Adaptively threshold attention weights to create sparse masks."""
        
        # Compute threshold per head and layer
        threshold = torch.quantile(attention_weights, percentile / 100.0, dim=-1, keepdim=True)
        
        # Create adaptive mask
        adaptive_mask = (attention_weights >= threshold).float()
        
        # Ensure minimum connectivity
        min_connections = max(1, attention_weights.size(-1) // 10)
        top_k_mask = torch.zeros_like(attention_weights)
        
        # Keep top-k connections for each query
        _, top_indices = torch.topk(attention_weights, min_connections, dim=-1)
        top_k_mask.scatter_(-1, top_indices, 1)
        
        # Combine adaptive and top-k masks
        final_mask = torch.maximum(adaptive_mask, top_k_mask)
        
        return final_mask
        
    def _create_cache_key(self, input_ids, mask_type):
        """Create cache key for mask caching."""
        # Simple hash based on sequence length and special token positions
        seq_len = input_ids.size(1)
        
        # Find special token positions
        special_positions = []
        special_tokens = [0, 1, 2, 3, 4]  # Common special token IDs
        
        for token_id in special_tokens:
            positions = (input_ids == token_id).nonzero(as_tuple=True)
            if len(positions[0]) > 0:
                special_positions.extend(positions[1].tolist())
                
        # Create hash
        cache_key = f"{mask_type}_{seq_len}_{hash(tuple(sorted(special_positions)))}"
        return cache_key