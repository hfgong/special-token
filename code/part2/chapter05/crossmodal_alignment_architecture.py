class CrossModalAlignmentLayer(nn.Module):
    def __init__(self, embed_dim=768, num_alignment_tokens=8):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_alignment_tokens = num_alignment_tokens
        
        # Learnable alignment tokens
        self.alignment_tokens = nn.Parameter(
            torch.randn(num_alignment_tokens, embed_dim)
        )
        
        # Cross-modal attention mechanisms
        self.cross_attention_v2t = nn.MultiheadAttention(
            embed_dim, num_heads=12, batch_first=True
        )
        self.cross_attention_t2v = nn.MultiheadAttention(
            embed_dim, num_heads=12, batch_first=True
        )
        self.cross_attention_a2vt = nn.MultiheadAttention(
            embed_dim, num_heads=12, batch_first=True
        )
        
        # Alignment scoring
        self.alignment_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        
        # Layer normalizations
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, visual_tokens, text_tokens, audio_tokens=None):
        batch_size = visual_tokens.shape[0]
        
        # Expand alignment tokens for batch
        alignment_tokens = self.alignment_tokens.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # Cross-modal alignment: visual to text
        aligned_v2t, attn_weights_v2t = self.cross_attention_v2t(
            query=alignment_tokens,
            key=torch.cat([visual_tokens, text_tokens], dim=1),
            value=torch.cat([visual_tokens, text_tokens], dim=1)
        )
        
        # Cross-modal alignment: text to visual
        aligned_t2v, attn_weights_t2v = self.cross_attention_t2v(
            query=alignment_tokens,
            key=torch.cat([text_tokens, visual_tokens], dim=1),
            value=torch.cat([text_tokens, visual_tokens], dim=1)
        )
        
        # Audio alignment if available
        if audio_tokens is not None:
            multimodal_tokens = torch.cat([visual_tokens, text_tokens, audio_tokens], dim=1)
            aligned_multimodal, _ = self.cross_attention_a2vt(
                query=alignment_tokens,
                key=multimodal_tokens,
                value=multimodal_tokens
            )
            alignment_tokens = alignment_tokens + aligned_multimodal
        
        # Combine alignments
        alignment_tokens = self.layer_norm1(
            alignment_tokens + aligned_v2t + aligned_t2v
        )
        
        # Compute alignment scores
        alignment_scores = []
        for i in range(self.num_alignment_tokens):
            token_features = alignment_tokens[:, i, :]  # [B, embed_dim]
            
            # Score against visual-text pairs
            vt_features = []
            for v_idx in range(visual_tokens.shape[1]):
                for t_idx in range(text_tokens.shape[1]):
                    v_feat = visual_tokens[:, v_idx, :]
                    t_feat = text_tokens[:, t_idx, :]
                    combined = torch.cat([v_feat, t_feat], dim=-1)
                    score = self.alignment_scorer(combined)
                    vt_features.append(score)
            
            if vt_features:
                alignment_scores.append(torch.stack(vt_features, dim=1))
        
        alignment_scores = torch.stack(alignment_scores, dim=1) if alignment_scores else None
        
        return {
            'alignment_tokens': alignment_tokens,
            'alignment_scores': alignment_scores,
            'attention_weights': {
                'v2t': attn_weights_v2t,
                't2v': attn_weights_t2v
            }
        }

class AlignedMultimodalTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=768):
        super().__init__()
        
        # Modality encoders
        self.text_encoder = nn.Embedding(vocab_size, embed_dim)
        self.visual_encoder = VisionTransformer(embed_dim=embed_dim)
        self.audio_encoder = AudioEncoder(embed_dim=embed_dim)
        
        # Alignment layers
        self.alignment_layers = nn.ModuleList([
            CrossModalAlignmentLayer(embed_dim) for _ in range(4)
        ])
        
        # Final fusion transformer
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=12,
                batch_first=True
            ),
            num_layers=6
        )
        
        # Task-specific heads
        self.classification_head = nn.Linear(embed_dim, vocab_size)
        self.retrieval_head = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, text_ids, images, audio_features=None, task='classification'):
        # Encode modalities
        text_tokens = self.text_encoder(text_ids)
        visual_tokens = self.visual_encoder(images)
        
        audio_tokens = None
        if audio_features is not None:
            audio_tokens = self.audio_encoder(audio_features)
        
        # Progressive alignment
        alignment_outputs = []
        for alignment_layer in self.alignment_layers:
            alignment_output = alignment_layer(visual_tokens, text_tokens, audio_tokens)
            alignment_outputs.append(alignment_output)
            
            # Update tokens with alignment information
            alignment_tokens = alignment_output['alignment_tokens']
            
            # Incorporate alignment back into modality representations
            text_tokens = text_tokens + alignment_tokens.mean(dim=1, keepdim=True)
            visual_tokens = visual_tokens + alignment_tokens.mean(dim=1, keepdim=True)
        
        # Combine all modalities with final alignment
        final_alignment = alignment_outputs[-1]['alignment_tokens']
        combined_tokens = torch.cat([
            text_tokens, visual_tokens, final_alignment
        ], dim=1)
        
        # Final fusion
        fused_output = self.fusion_transformer(combined_tokens)
        
        # Task-specific processing
        if task == 'classification':
            # Use first token for classification
            output = self.classification_head(fused_output[:, 0])
        elif task == 'retrieval':
            # Pool for retrieval
            pooled = fused_output.mean(dim=1)
            output = self.retrieval_head(pooled)
        else:
            output = fused_output
        
        return {
            'output': output,
            'alignment_outputs': alignment_outputs,
            'fused_representation': fused_output
        }
