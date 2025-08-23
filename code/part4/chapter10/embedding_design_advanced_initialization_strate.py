"""
Advanced initialization strategies for special token embeddings

Extracted from: part4/chapter10/embedding_design.tex
Block: 1
Lines: 106
"""

import torch
import torch.nn as nn
import numpy as np

class SpecialTokenEmbeddingInitializer:
    def __init__(self, model, embedding_dim=768):
        self.model = model
        self.embedding_dim = embedding_dim
        self.existing_embeddings = model.embeddings.word_embeddings.weight.data
        
    def initialize_special_tokens(self, special_token_ids, strategy='xavier_uniform'):
        """Initialize special token embeddings with various strategies."""
        
        for token_id in special_token_ids:
            if strategy == 'xavier_uniform':
                embedding = self._xavier_uniform_init()
            elif strategy == 'xavier_normal':
                embedding = self._xavier_normal_init()
            elif strategy == 'average_existing':
                embedding = self._average_existing_init()
            elif strategy == 'contextual_similarity':
                embedding = self._contextual_similarity_init(token_id)
            elif strategy == 'task_specific':
                embedding = self._task_specific_init(token_id)
            elif strategy == 'orthogonal':
                embedding = self._orthogonal_init()
            else:
                raise ValueError(f"Unknown initialization strategy: {strategy}")
                
            self.model.embeddings.word_embeddings.weight.data[token_id] = embedding
            
    def _xavier_uniform_init(self):
        """Xavier uniform initialization."""
        limit = np.sqrt(6.0 / (self.embedding_dim + 1))
        return torch.FloatTensor(self.embedding_dim).uniform_(-limit, limit)
        
    def _xavier_normal_init(self):
        """Xavier normal initialization."""
        std = np.sqrt(2.0 / (self.embedding_dim + 1))
        return torch.randn(self.embedding_dim) * std
        
    def _average_existing_init(self):
        """Initialize as average of existing embeddings."""
        # Sample random subset to avoid memory issues
        num_samples = min(1000, len(self.existing_embeddings))
        indices = torch.randperm(len(self.existing_embeddings))[:num_samples]
        sampled_embeddings = self.existing_embeddings[indices]
        return sampled_embeddings.mean(dim=0)
        
    def _contextual_similarity_init(self, token_id):
        """Initialize based on contextual similarity to token purpose."""
        # Map special tokens to similar existing tokens
        similarity_map = {
            '[CLS]': ['start', 'begin', 'first'],
            '[SEP]': ['separator', 'divide', 'split'],
            '[MASK]': ['unknown', 'hidden', 'blank'],
            '[PAD]': ['padding', 'fill', 'empty'],
        }
        
        # Get token string
        token_str = self.model.tokenizer.convert_ids_to_tokens([token_id])[0]
        
        # Find similar tokens
        similar_tokens = similarity_map.get(token_str, [])
        if similar_tokens:
            similar_ids = self.model.tokenizer.convert_tokens_to_ids(similar_tokens)
            similar_embeddings = self.existing_embeddings[similar_ids]
            return similar_embeddings.mean(dim=0)
        else:
            return self._average_existing_init()
            
    def _task_specific_init(self, token_id):
        """Initialize based on intended task."""
        token_str = self.model.tokenizer.convert_ids_to_tokens([token_id])[0]
        
        if '[CLS]' in token_str:
            # Initialize for classification: slight bias toward positive dimensions
            base = self._xavier_normal_init()
            base[:self.embedding_dim//2] *= 1.1
            return base
        elif '[SEP]' in token_str:
            # Initialize for separation: orthogonal to average
            avg = self._average_existing_init()
            orthogonal = self._make_orthogonal_to(avg)
            return orthogonal
        elif '[MASK]' in token_str:
            # Initialize for masking: closer to uniform distribution
            return torch.randn(self.embedding_dim) * 0.02
        else:
            return self._xavier_uniform_init()
            
    def _orthogonal_init(self):
        """Initialize orthogonal to existing embeddings."""
        # Use QR decomposition to find orthogonal vector
        sample_embeddings = self.existing_embeddings[:min(100, len(self.existing_embeddings))]
        Q, _ = torch.qr(sample_embeddings.T)
        
        # Take a column that's orthogonal to existing space
        if Q.shape[1] < self.embedding_dim:
            # Find orthogonal complement
            return self._find_orthogonal_complement(Q)
        else:
            # Use last column as it's most orthogonal
            return Q[:, -1]
            
    def _make_orthogonal_to(self, vector):
        """Make a random vector orthogonal to given vector."""
        random_vec = torch.randn_like(vector)
        # Gram-Schmidt process
        projection = (random_vec @ vector) / (vector @ vector) * vector
        orthogonal = random_vec - projection
        return orthogonal / orthogonal.norm()
        
    def _find_orthogonal_complement(self, Q):
        """Find vector in orthogonal complement of Q."""
        # Create random vector
        v = torch.randn(self.embedding_dim)
        
        # Project out components in Q
        for i in range(Q.shape[1]):
            q_i = Q[:, i]
            v = v - (v @ q_i) * q_i
            
        return v / v.norm()