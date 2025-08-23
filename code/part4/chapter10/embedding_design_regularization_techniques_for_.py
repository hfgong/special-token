"""
Regularization techniques for special token embeddings

Extracted from: part4/chapter10/embedding_design.tex
Block: 3
Lines: 67
"""

class EmbeddingRegularizer:
    def __init__(self, model, special_token_ids, reg_weight=0.01):
        self.model = model
        self.special_token_ids = special_token_ids
        self.reg_weight = reg_weight
        self.reference_embeddings = None
        
    def initialize_references(self):
        """Store reference embeddings for regularization."""
        embeddings = self.model.embeddings.word_embeddings.weight.data
        self.reference_embeddings = embeddings.clone()
        
    def l2_regularization(self):
        """L2 regularization to prevent large deviations."""
        embeddings = self.model.embeddings.word_embeddings.weight
        reg_loss = 0
        
        for token_id in self.special_token_ids:
            current_emb = embeddings[token_id]
            reference_emb = self.reference_embeddings[token_id]
            
            # L2 distance from reference
            reg_loss += torch.norm(current_emb - reference_emb, p=2) ** 2
            
        return self.reg_weight * reg_loss
        
    def cosine_similarity_regularization(self):
        """Maintain cosine similarity with neighboring embeddings."""
        embeddings = self.model.embeddings.word_embeddings.weight
        reg_loss = 0
        
        for token_id in self.special_token_ids:
            special_emb = embeddings[token_id]
            
            # Sample neighboring embeddings
            num_neighbors = 10
            neighbor_ids = torch.randperm(len(embeddings))[:num_neighbors]
            neighbor_embs = embeddings[neighbor_ids]
            
            # Compute average cosine similarity
            cosine_sims = torch.nn.functional.cosine_similarity(
                special_emb.unsqueeze(0),
                neighbor_embs,
                dim=1
            )
            
            # Regularize to maintain moderate similarity (not too high, not too low)
            target_similarity = 0.3
            reg_loss += ((cosine_sims - target_similarity) ** 2).mean()
            
        return self.reg_weight * reg_loss
        
    def spectral_regularization(self):
        """Regularize spectral properties of embedding matrix."""
        embeddings = self.model.embeddings.word_embeddings.weight
        
        # Include special tokens in spectral analysis
        special_embeddings = embeddings[self.special_token_ids]
        
        # Compute singular values
        _, S, _ = torch.svd(special_embeddings)
        
        # Regularize condition number (ratio of largest to smallest singular value)
        condition_number = S[0] / (S[-1] + 1e-8)
        
        # Penalty for high condition number
        reg_loss = self.reg_weight * torch.log(condition_number)
        
        return reg_loss
        
    def diversity_regularization(self):
        """Encourage diversity among special token embeddings."""
        embeddings = self.model.embeddings.word_embeddings.weight
        special_embeddings = embeddings[self.special_token_ids]
        
        # Compute pairwise similarities
        similarities = torch.mm(special_embeddings, special_embeddings.T)
        
        # Normalize by embedding norms
        norms = torch.norm(special_embeddings, dim=1, keepdim=True)
        norm_matrix = torch.mm(norms, norms.T)
        similarities = similarities / (norm_matrix + 1e-8)
        
        # Penalty for high similarity (encourage diversity)
        # Exclude diagonal (self-similarity)
        mask = 1 - torch.eye(len(special_embeddings), device=similarities.device)
        reg_loss = (similarities * mask).abs().mean()
        
        return self.reg_weight * reg_loss