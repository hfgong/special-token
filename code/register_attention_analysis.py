def visualize_register_attention(model, image, layer_idx=-1):
    """Visualize how register tokens attend to image patches."""
    model.eval()
    
    with torch.no_grad():
        # Get attention weights
        output = model(image.unsqueeze(0), output_attentions=True)
        attention = output.attentions[layer_idx][0]  # [num_heads, seq_len, seq_len]
        
        # Extract register token attention patterns
        num_register_tokens = model.num_register_tokens
        register_start_idx = 1  # After CLS token
        register_end_idx = register_start_idx + num_register_tokens
        
        # Attention from register tokens to patches
        patch_start_idx = register_end_idx
        register_to_patch = attention[:, register_start_idx:register_end_idx, patch_start_idx:]
        
        # Average across heads
        avg_attention = register_to_patch.mean(dim=0)  # [num_registers, num_patches]
        
        # Reshape to spatial grid for visualization
        H = W = int(math.sqrt(avg_attention.shape[1]))
        spatial_attention = avg_attention.view(num_register_tokens, H, W)
        
        return spatial_attention

def plot_register_attention_maps(spatial_attention, image):
    """Plot attention maps for each register token."""
    num_registers = spatial_attention.shape[0]
    
    fig, axes = plt.subplots(2, (num_registers + 1) // 2 + 1, figsize=(15, 8))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Register token attention maps
    for i in range(num_registers):
        ax = axes[i + 1]
        attention_map = spatial_attention[i].cpu().numpy()
        
        im = ax.imshow(attention_map, cmap='hot', interpolation='bilinear')
        ax.set_title(f'Register Token {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(num_registers + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()