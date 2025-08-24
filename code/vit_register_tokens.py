class ViTWithRegisterTokens(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, 
                 num_register_tokens=4, num_classes=1000):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        
        # Special tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(
            torch.zeros(1, num_register_tokens, embed_dim)
        )
        
        # Position embeddings for all tokens
        total_tokens = 1 + num_register_tokens + self.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, total_tokens, embed_dim)
        )
        
        self.transformer = TransformerEncoder(embed_dim, num_layers=12)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize tokens
        self._init_tokens()
    
    def _init_tokens(self):
        """Initialize special tokens with appropriate distributions."""
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        torch.nn.init.trunc_normal_(self.register_tokens, std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Expand special tokens for batch
        cls_tokens = self.cls_token.expand(B, -1, -1)
        register_tokens = self.register_tokens.expand(B, -1, -1)
        
        # Concatenate all tokens: [CLS] + [REG_1, REG_2, ...] + patches
        x = torch.cat([cls_tokens, register_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Transformer processing
        x = self.transformer(x)
        
        # Extract CLS token for classification (register tokens ignored)
        cls_output = x[:, 0]
        
        return self.head(cls_output)