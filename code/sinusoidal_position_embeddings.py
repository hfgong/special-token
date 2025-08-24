def get_2d_sincos_pos_embed(grid_size, embed_dim, temperature=10000):
    """
    Generate 2D sinusoidal position embeddings
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h, indexing='xy')
    grid = np.stack(grid, axis=0)  # [2, grid_size, grid_size]
    
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Generate sinusoidal embeddings from 2D grid coordinates"""
    assert embed_dim % 2 == 0
    
    # Use half of dimensions for each axis
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # H
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # W
    
    emb = np.concatenate([emb_h, emb_w], axis=1)  # [H*W, embed_dim]
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sinusoidal embeddings"""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # [embed_dim//2,]
    
    pos = pos.reshape(-1)  # [M,]
    out = np.einsum('m,d->md', pos, omega)  # [M, embed_dim//2], outer product
    
    emb_sin = np.sin(out)  # [M, embed_dim//2]
    emb_cos = np.cos(out)  # [M, embed_dim//2]
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # [M, embed_dim]
    return emb

class SinCos2DPositionEmbedding(nn.Module):
    def __init__(self, embed_dim=768, temperature=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
    
    def forward(self, x, grid_size):
        pos_embed = get_2d_sincos_pos_embed(grid_size, self.embed_dim, self.temperature)
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
        
        # Add CLS position (zeros)
        cls_pos_embed = torch.zeros(1, 1, self.embed_dim)
        pos_embed = torch.cat([cls_pos_embed, pos_embed], dim=1)
        
        return x + pos_embed.to(x.device)