class DynamicRegisterViT(nn.Module):
    def __init__(self, embed_dim=768, max_register_tokens=8):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_register_tokens = max_register_tokens
        
        # Pool of register tokens
        self.register_token_pool = nn.Parameter(
            torch.zeros(1, max_register_tokens, embed_dim)
        )
        
        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def select_register_tokens(self, patch_embeddings):
        """Dynamically select number of register tokens based on input."""
        # Estimate input complexity
        complexity = self.complexity_estimator(
            patch_embeddings.mean(dim=1)  # Global average
        ).squeeze(-1)  # [B]
        
        # Scale to number of tokens
        num_tokens = (complexity * self.max_register_tokens).round().long()
        
        # Ensure at least one token
        num_tokens = torch.clamp(num_tokens, min=1, max=self.max_register_tokens)
        
        return num_tokens
    
    def forward(self, patch_embeddings):
        B = patch_embeddings.shape[0]
        
        # Determine register token allocation
        num_register_tokens = self.select_register_tokens(patch_embeddings)
        
        # Create batch-specific register tokens
        register_tokens_list = []
        for b in range(B):
            n_tokens = num_register_tokens[b].item()
            batch_registers = self.register_token_pool[:, :n_tokens, :].expand(1, -1, -1)
            register_tokens_list.append(batch_registers)
        
        # Pad to maximum length for batching
        max_tokens = num_register_tokens.max().item()
        padded_registers = torch.zeros(B, max_tokens, self.embed_dim, 
                                     device=patch_embeddings.device)
        
        for b, tokens in enumerate(register_tokens_list):
            padded_registers[b, :tokens.shape[1], :] = tokens
        
        return padded_registers, num_register_tokens