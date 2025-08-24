def interpolate_pos_embed(pos_embed, orig_size, new_size):
    """
    Interpolate position embeddings for different image sizes
    
    Args:
        pos_embed: [1, N+1, D] where N = orig_size^2
        orig_size: Original grid size (e.g., 14 for 224x224 with 16x16 patches)
        new_size: Target grid size
    """
    # Extract CLS and patch position embeddings
    cls_pos_embed = pos_embed[:, 0:1]
    patch_pos_embed = pos_embed[:, 1:]
    
    if orig_size == new_size:
        return pos_embed
    
    # Reshape patch embeddings to 2D grid
    embed_dim = patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, embed_dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # [1, D, H, W]
    
    # Interpolate to new size
    patch_pos_embed_resized = F.interpolate(
        patch_pos_embed,
        size=(new_size, new_size),
        mode='bicubic',
        align_corners=False
    )
    
    # Reshape back to sequence format
    patch_pos_embed_resized = patch_pos_embed_resized.permute(0, 2, 3, 1)
    patch_pos_embed_resized = patch_pos_embed_resized.reshape(1, new_size**2, embed_dim)
    
    # Concatenate CLS and interpolated patch embeddings
    pos_embed_resized = torch.cat([cls_pos_embed, patch_pos_embed_resized], dim=1)
    
    return pos_embed_resized

def adaptive_pos_embed(model, image_size):
    """Adapt model's position embeddings to new image size"""
    
    # Calculate new grid size
    patch_size = model.patch_embed.patch_size
    new_grid_size = image_size // patch_size
    orig_grid_size = int(math.sqrt(model.pos_embed.shape[1] - 1))
    
    if new_grid_size != orig_grid_size:
        # Interpolate position embeddings
        new_pos_embed = interpolate_pos_embed(
            model.pos_embed.data,
            orig_grid_size,
            new_grid_size
        )
        
        # Update model's position embeddings
        model.pos_embed = nn.Parameter(new_pos_embed)
    
    return model