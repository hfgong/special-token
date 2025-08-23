"""
Advanced attention masking patterns

Extracted from: part4/chapter10/attention_masks.tex
Block: 2
Lines: 111
"""

class AdvancedMaskingPatterns:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def create_hierarchical_mask(self, input_ids, segment_ids=None):
        """Create hierarchical attention masks for structured inputs."""
        batch_size, seq_len = input_ids.shape
        
        # Base attention mask
        attention_mask = torch.ones(batch_size, seq_len, seq_len)
        
        if segment_ids is not None:
            # Within-segment attention
            for batch_idx in range(batch_size):
                for i in range(seq_len):
                    for j in range(seq_len):
                        # Allow attention within same segment
                        if segment_ids[batch_idx, i] == segment_ids[batch_idx, j]:
                            attention_mask[batch_idx, i, j] = 1
                        else:
                            attention_mask[batch_idx, i, j] = 0
                            
        # Special token override rules
        cls_token_id = getattr(self.tokenizer, 'cls_token_id', None)
        sep_token_id = getattr(self.tokenizer, 'sep_token_id', None)
        
        for batch_idx in range(batch_size):
            # CLS token can attend to everything
            if cls_token_id is not None:
                cls_positions = (input_ids[batch_idx] == cls_token_id).nonzero(as_tuple=True)[0]
                for pos in cls_positions:
                    attention_mask[batch_idx, pos, :] = 1
                    attention_mask[batch_idx, :, pos] = 1
                    
            # SEP tokens have limited attention
            if sep_token_id is not None:
                sep_positions = (input_ids[batch_idx] == sep_token_id).nonzero(as_tuple=True)[0]
                for pos in sep_positions:
                    # SEP only attends to segment boundaries
                    attention_mask[batch_idx, pos, :] = 0
                    attention_mask[batch_idx, pos, sep_positions] = 1
                    if cls_token_id is not None:
                        cls_positions = (input_ids[batch_idx] == cls_token_id).nonzero(as_tuple=True)[0]
                        attention_mask[batch_idx, pos, cls_positions] = 1
                        
        return attention_mask.unsqueeze(1).float()
        
    def create_sliding_window_mask(self, input_ids, window_size=128, special_token_global=True):
        """Create sliding window attention with global special tokens."""
        batch_size, seq_len = input_ids.shape
        
        # Initialize with zeros
        attention_mask = torch.zeros(batch_size, seq_len, seq_len)
        
        # Apply sliding window
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            attention_mask[:, i, start:end] = 1
            
        if special_token_global:
            # Special tokens have global attention
            special_tokens = [
                getattr(self.tokenizer, 'cls_token_id', None),
                getattr(self.tokenizer, 'sep_token_id', None),
            ]
            
            for batch_idx in range(batch_size):
                for token_id in special_tokens:
                    if token_id is not None:
                        special_positions = (input_ids[batch_idx] == token_id).nonzero(as_tuple=True)[0]
                        for pos in special_positions:
                            attention_mask[batch_idx, pos, :] = 1
                            attention_mask[batch_idx, :, pos] = 1
                            
        # Apply padding mask
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        if pad_token_id is not None:
            padding_mask = (input_ids != pad_token_id)
            attention_mask = attention_mask * padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)
            
        return attention_mask.unsqueeze(1).float()
        
    def create_sparse_attention_mask(self, input_ids, sparsity_pattern='block_sparse'):
        """Create sparse attention patterns for efficiency."""
        batch_size, seq_len = input_ids.shape
        
        if sparsity_pattern == 'block_sparse':
            mask = self._create_block_sparse_mask(seq_len, block_size=64)
        elif sparsity_pattern == 'strided':
            mask = self._create_strided_mask(seq_len, stride=4)
        elif sparsity_pattern == 'random':
            mask = self._create_random_sparse_mask(seq_len, density=0.1)
        else:
            raise ValueError(f"Unknown sparsity pattern: {sparsity_pattern}")
            
        # Ensure special tokens have full attention
        cls_token_id = getattr(self.tokenizer, 'cls_token_id', None)
        
        for batch_idx in range(batch_size):
            if cls_token_id is not None:
                cls_positions = (input_ids[batch_idx] == cls_token_id).nonzero(as_tuple=True)[0]
                for pos in cls_positions:
                    mask[pos, :] = 1
                    mask[:, pos] = 1
                    
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).float()
        
    def _create_block_sparse_mask(self, seq_len, block_size=64):
        """Create block-sparse attention mask."""
        mask = torch.zeros(seq_len, seq_len)
        
        for i in range(0, seq_len, block_size):
            for j in range(0, seq_len, block_size):
                end_i = min(i + block_size, seq_len)
                end_j = min(j + block_size, seq_len)
                
                # Diagonal blocks
                if abs(i - j) <= block_size:
                    mask[i:end_i, j:end_j] = 1
                    
        return mask
        
    def _create_strided_mask(self, seq_len, stride=4):
        """Create strided attention mask."""
        mask = torch.zeros(seq_len, seq_len)
        
        for i in range(seq_len):
            # Local attention
            start = max(0, i - stride)
            end = min(seq_len, i + stride + 1)
            mask[i, start:end] = 1
            
            # Strided attention
            for j in range(0, seq_len, stride):
                mask[i, j] = 1
                
        return mask