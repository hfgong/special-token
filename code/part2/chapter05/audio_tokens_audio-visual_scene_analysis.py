"""
Audio-visual scene analysis

Extracted from: part2/chapter05/audio_tokens.tex
Block: 6
Lines: 59
"""

class AudioVisualSceneAnalyzer(nn.Module):
    def __init__(self, num_audio_classes=50, num_visual_classes=100, 
                 num_scene_classes=25, embed_dim=768):
        super().__init__()
        
        self.multimodal_transformer = AudioVisualTextTransformer(
            vocab_size=10000, embed_dim=embed_dim
        )
        
        # Classification heads
        self.audio_classifier = nn.Linear(embed_dim, num_audio_classes)
        self.visual_classifier = nn.Linear(embed_dim, num_visual_classes)
        self.scene_classifier = nn.Linear(embed_dim * 2, num_scene_classes)
        
        # Feature aggregation
        self.audio_pool = nn.AdaptiveAvgPool1d(1)
        self.visual_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, audio_features, images, audio_labels=None, 
                visual_labels=None, scene_labels=None):
        # Process multimodal input
        outputs = self.multimodal_transformer(
            text_ids=torch.zeros(audio_features.shape[0], 1, dtype=torch.long),
            audio_features=audio_features,
            images=images
        )
        
        hidden_states = outputs['hidden_states']
        modality_labels = outputs['modality_labels']
        
        # Separate audio and visual representations
        audio_mask = (modality_labels == 1)
        visual_mask = (modality_labels == 2)
        
        # Pool audio features
        audio_features_pooled = None
        if audio_mask.any():
            audio_hidden = hidden_states[audio_mask.unsqueeze(-1).expand_as(hidden_states)]
            audio_hidden = audio_hidden.view(hidden_states.shape[0], -1, hidden_states.shape[-1])
            audio_features_pooled = self.audio_pool(audio_hidden.transpose(1, 2)).squeeze(-1)
        
        # Pool visual features
        visual_features_pooled = None
        if visual_mask.any():
            visual_hidden = hidden_states[visual_mask.unsqueeze(-1).expand_as(hidden_states)]
            visual_hidden = visual_hidden.view(hidden_states.shape[0], -1, hidden_states.shape[-1])
            visual_features_pooled = self.visual_pool(visual_hidden.transpose(1, 2)).squeeze(-1)
        
        # Classify individual modalities
        audio_logits = self.audio_classifier(audio_features_pooled) if audio_features_pooled is not None else None
        visual_logits = self.visual_classifier(visual_features_pooled) if visual_features_pooled is not None else None
        
        # Joint scene classification
        joint_features = torch.cat([audio_features_pooled, visual_features_pooled], dim=-1)
        scene_logits = self.scene_classifier(joint_features)
        
        # Compute losses if labels provided
        losses = {}
        if audio_labels is not None and audio_logits is not None:
            losses['audio_loss'] = F.cross_entropy(audio_logits, audio_labels)
        if visual_labels is not None and visual_logits is not None:
            losses['visual_loss'] = F.cross_entropy(visual_logits, visual_labels)
        if scene_labels is not None:
            losses['scene_loss'] = F.cross_entropy(scene_logits, scene_labels)
        
        return {
            'audio_logits': audio_logits,
            'visual_logits': visual_logits,
            'scene_logits': scene_logits,
            'losses': losses
        }