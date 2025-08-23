"""
Multimodal transformer with audio token integration

Extracted from: part2/chapter05/audio_tokens.tex
Block: 3
Lines: 109
"""

class AudioVisualTextTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, audio_input_dim=105):
        super().__init__()
        
        # Modality-specific encoders
        self.text_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.audio_encoder = AudioEncoder(audio_input_dim, embed_dim)
        self.image_encoder = ImageEncoder(embed_dim)
        
        # Special token embeddings
        self.audio_token = nn.Parameter(torch.randn(1, embed_dim))
        self.img_token = nn.Parameter(torch.randn(1, embed_dim))
        
        # Cross-modal attention layers
        self.cross_modal_layers = nn.ModuleList([
            CrossModalAttentionLayer(embed_dim) for _ in range(6)
        ])
        
        # Final transformer layers
        self.final_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=12,
                batch_first=True
            ),
            num_layers=6
        )
        
        # Output heads
        self.classification_head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, text_ids, audio_features=None, images=None, 
                attention_mask=None):
        batch_size = text_ids.shape[0]
        
        # Process text
        text_embeds = self.text_embeddings(text_ids)
        
        # Initialize multimodal sequence with text
        multimodal_sequence = [text_embeds]
        modality_types = [torch.zeros(text_embeds.shape[:2], dtype=torch.long)]
        
        # Add audio if provided
        if audio_features is not None:
            audio_embeds = self.audio_encoder(audio_features)
            
            # Add audio token markers
            audio_markers = self.audio_token.expand(
                batch_size, audio_embeds.shape[1], -1
            )
            audio_embeds = audio_embeds + audio_markers
            
            multimodal_sequence.append(audio_embeds)
            modality_types.append(torch.ones(audio_embeds.shape[:2], dtype=torch.long))
        
        # Add images if provided
        if images is not None:
            image_embeds = self.image_encoder(images)
            
            # Add image token markers
            image_markers = self.img_token.expand(
                batch_size, image_embeds.shape[1], -1
            )
            image_embeds = image_embeds + image_markers
            
            multimodal_sequence.append(image_embeds)
            modality_types.append(torch.full(image_embeds.shape[:2], 2, dtype=torch.long))
        
        # Concatenate all modalities
        full_sequence = torch.cat(multimodal_sequence, dim=1)
        modality_labels = torch.cat(modality_types, dim=1)
        
        # Cross-modal processing
        for layer in self.cross_modal_layers:
            full_sequence = layer(full_sequence, modality_labels)
        
        # Final transformer processing
        output = self.final_transformer(full_sequence)
        
        # Classification
        logits = self.classification_head(output)
        
        return {
            'logits': logits,
            'hidden_states': output,
            'modality_labels': modality_labels
        }

class CrossModalAttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads=12, batch_first=True
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads=12, batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, modality_labels):
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # Cross-modal attention (audio attending to text/image)
        audio_mask = (modality_labels == 1)
        if audio_mask.any():
            audio_tokens = x[audio_mask.unsqueeze(-1).expand_as(x)].view(
                x.shape[0], -1, x.shape[-1]
            )
            other_tokens = x[~audio_mask.unsqueeze(-1).expand_as(x)].view(
                x.shape[0], -1, x.shape[-1]
            )
            
            if other_tokens.shape[1] > 0:
                cross_attn_output, _ = self.cross_attention(
                    audio_tokens, other_tokens, other_tokens
                )
                # Update audio tokens with cross-modal information
                x[audio_mask.unsqueeze(-1).expand_as(x)] = cross_attn_output.flatten()
        
        x = self.layer_norm2(x)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.layer_norm3(x + ff_output)
        
        return x