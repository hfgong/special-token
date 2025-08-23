"""
Video frame token architecture

Extracted from: part2/chapter05/video_tokens.tex
Block: 1
Lines: 78
"""

class VideoFrameEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_frames=16, frame_size=224):
        super().__init__()
        
        self.num_frames = num_frames
        
        # Per-frame spatial encoder (Vision Transformer)
        self.frame_encoder = VisionTransformer(
            image_size=frame_size,
            patch_size=16,
            embed_dim=embed_dim
        )
        
        # Temporal attention across frames
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=12,
            batch_first=True
        )
        
        # Temporal position embeddings
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, num_frames, embed_dim)
        )
        
        # Video token summarization
        self.video_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, video_frames):
        # video_frames: [B, T, C, H, W]
        batch_size, num_frames, c, h, w = video_frames.shape
        
        # Process each frame independently
        frame_features = []
        for t in range(num_frames):
            frame_feat = self.frame_encoder(video_frames[:, t])  # [B, num_patches, embed_dim]
            # Use CLS token as frame representation
            frame_features.append(frame_feat[:, 0])  # [B, embed_dim]
        
        # Stack temporal features
        temporal_features = torch.stack(frame_features, dim=1)  # [B, T, embed_dim]
        
        # Add temporal position embeddings
        temporal_features = temporal_features + self.temporal_pos_embed[:, :num_frames]
        
        # Temporal attention processing
        video_tokens = self.video_token.expand(batch_size, -1, -1)
        video_representation, _ = self.temporal_attention(
            query=video_tokens,
            key=temporal_features,
            value=temporal_features
        )
        
        return video_representation, temporal_features

class VideoTextTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=768):
        super().__init__()
        
        self.text_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.video_encoder = VideoFrameEncoder(embed_dim)
        
        # Video token marker
        self.video_token_marker = nn.Parameter(torch.randn(1, embed_dim))
        
        # Multimodal transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=12,
                batch_first=True
            ),
            num_layers=12
        )
        
        # Output heads
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, text_ids, video_frames=None):
        # Process text
        text_embeds = self.text_embeddings(text_ids)
        
        if video_frames is not None:
            # Process video
            video_repr, _ = self.video_encoder(video_frames)
            
            # Add video token marker
            video_repr = video_repr + self.video_token_marker
            
            # Combine text and video
            combined_embeds = torch.cat([video_repr, text_embeds], dim=1)
        else:
            combined_embeds = text_embeds
        
        # Transformer processing
        output = self.transformer(combined_embeds)
        
        # Language modeling
        logits = self.lm_head(output)
        
        return logits