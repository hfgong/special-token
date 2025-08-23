"""
Register token regularization strategies

Extracted from: part2/chapter04/register_tokens.tex
Block: 4
Lines: 55
"""

class RegisterTokenRegularizer:
    def __init__(self, diversity_weight=0.01, sparsity_weight=0.001):
        self.diversity_weight = diversity_weight
        self.sparsity_weight = sparsity_weight
    
    def diversity_loss(self, register_tokens):
        """Encourage diversity among register tokens."""
        # register_tokens: [B, num_registers, embed_dim]
        B, N, D = register_tokens.shape
        
        # Compute pairwise similarities
        normalized_tokens = F.normalize(register_tokens, dim=-1)
        similarity_matrix = torch.bmm(normalized_tokens, normalized_tokens.transpose(-2, -1))
        
        # Penalize high off-diagonal similarities
        identity = torch.eye(N, device=register_tokens.device).unsqueeze(0).expand(B, -1, -1)
        off_diagonal = similarity_matrix * (1 - identity)
        
        diversity_loss = off_diagonal.abs().mean()
        return diversity_loss
    
    def sparsity_loss(self, attention_weights, register_indices):
        """Encourage sparse attention to register tokens."""
        # attention_weights: [B, num_heads, seq_len, seq_len]
        # register_indices: indices of register tokens in sequence
        
        B, H, S, _ = attention_weights.shape
        
        # Extract attention to register tokens
        register_attention = attention_weights[:, :, :, register_indices]
        
        # L1 sparsity penalty
        sparsity_loss = register_attention.abs().mean()
        return sparsity_loss
    
    def compute_regularization(self, register_tokens, attention_weights, register_indices):
        """Compute total regularization loss."""
        div_loss = self.diversity_loss(register_tokens)
        sparse_loss = self.sparsity_loss(attention_weights, register_indices)
        
        total_reg = (self.diversity_weight * div_loss + 
                    self.sparsity_weight * sparse_loss)
        
        return total_reg, {'diversity': div_loss, 'sparsity': sparse_loss}

# Usage in training loop
regularizer = RegisterTokenRegularizer()

def training_step(model, batch, optimizer):
    output, attention_weights = model(batch, return_attention=True)
    
    # Main task loss
    task_loss = F.cross_entropy(output, batch.targets)
    
    # Register token regularization
    register_tokens = model.get_register_representations()
    register_indices = list(range(1, 1 + model.num_register_tokens))
    
    reg_loss, reg_components = regularizer.compute_regularization(
        register_tokens, attention_weights, register_indices
    )
    
    # Total loss
    total_loss = task_loss + reg_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return {
        'task_loss': task_loss.item(),
        'reg_loss': reg_loss.item(),
        **{f'reg_{k}': v.item() for k, v in reg_components.items()}
    }