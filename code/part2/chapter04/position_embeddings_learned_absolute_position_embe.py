"""
Learned absolute position embeddings

Extracted from: part2/chapter04/position_embeddings.tex
Block: 1
Lines: 51
"""

class LearnedPositionEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        # Learnable position embeddings for each patch position
        # +1 for CLS token
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
    
    def forward(self, x):
        # x shape: [batch_size, num_patches + 1, embed_dim]
        return x + self.pos_embed

class AdaptivePositionEmbedding(nn.Module):
    def __init__(self, max_grid_size=32, embed_dim=768):
        super().__init__()
        
        self.max_grid_size = max_grid_size
        self.embed_dim = embed_dim
        
        # Create position embeddings for maximum possible grid
        self.pos_embed_cache = nn.Parameter(
            torch.randn(1, max_grid_size**2 + 1, embed_dim) * 0.02
        )
    
    def interpolate_pos_embed(self, grid_size):
        """Interpolate position embeddings for different image sizes"""
        
        if grid_size == self.max_grid_size:
            return self.pos_embed_cache
        
        # Extract patch embeddings (excluding CLS)
        pos_embed_patches = self.pos_embed_cache[:, 1:]
        
        # Reshape to 2D grid for interpolation
        pos_embed_2d = pos_embed_patches.view(
            1, self.max_grid_size, self.max_grid_size, self.embed_dim
        ).permute(0, 3, 1, 2)
        
        # Interpolate to target grid size
        pos_embed_resized = F.interpolate(
            pos_embed_2d, 
            size=(grid_size, grid_size), 
            mode='bicubic',
            align_corners=False
        )
        
        # Reshape back to sequence format
        pos_embed_resized = pos_embed_resized.permute(0, 2, 3, 1).view(
            1, grid_size**2, self.embed_dim
        )
        
        # Concatenate with CLS position embedding
        cls_pos_embed = self.pos_embed_cache[:, :1]
        
        return torch.cat([cls_pos_embed, pos_embed_resized], dim=1)
    
    def forward(self, x, grid_size):
        pos_embed = self.interpolate_pos_embed(grid_size)
        return x + pos_embed