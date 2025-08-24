class HierarchicalPositionEmbedding(nn.Module):
    def __init__(self, embed_dims=[96, 192, 384, 768], grid_sizes=[56, 28, 14, 7]):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.grid_sizes = grid_sizes
        self.num_stages = len(embed_dims)
        
        # Position embeddings for each stage
        self.pos_embeds = nn.ModuleList([
            nn.Parameter(torch.randn(1, grid_sizes[i]**2, embed_dims[i]) * 0.02)
            for i in range(self.num_stages)
        ])
        
        # Cross-scale position alignment
        self.scale_aligners = nn.ModuleList([
            nn.Linear(embed_dims[i], embed_dims[i+1])
            for i in range(self.num_stages - 1)
        ])
    
    def forward(self, features_list):
        """
        features_list: List of features at different scales
        """
        enhanced_features = []
        
        for i, features in enumerate(features_list):
            # Add position embeddings for current scale
            pos_embed = self.pos_embeds[i]
            features_with_pos = features + pos_embed
            
            # Cross-scale position information
            if i > 0:
                # Get position information from previous scale
                prev_pos = enhanced_features[i-1]
                
                # Downsample and align dimensions
                prev_pos_downsampled = F.adaptive_avg_pool1d(
                    prev_pos.transpose(1, 2), 
                    self.grid_sizes[i]**2
                ).transpose(1, 2)
                
                prev_pos_aligned = self.scale_aligners[i-1](prev_pos_downsampled)
                
                # Combine current and previous scale position information
                features_with_pos = features_with_pos + 0.1 * prev_pos_aligned
            
            enhanced_features.append(features_with_pos)
        
        return enhanced_features