"""
Robust classification with modality switching

Extracted from: part2/chapter05/modality_switching.tex
Block: 2
Lines: 63
"""

class RobustMultimodalClassifier(nn.Module):
    def __init__(self, num_classes, embed_dim=768):
        super().__init__()
        
        self.adaptive_model = AdaptiveMultimodalTransformer(
            vocab_size=30000, embed_dim=embed_dim
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, text_ids=None, images=None, audio_features=None):
        # Adaptive multimodal processing
        outputs = self.adaptive_model(
            text_ids=text_ids,
            images=images,
            audio_features=audio_features,
            task='classification'
        )
        
        # Classification
        logits = self.classifier(outputs['final_representation'])
        
        # Confidence estimation
        confidence = self.confidence_estimator(outputs['final_representation'])
        
        return {
            'logits': logits,
            'confidence': confidence,
            'modality_importance': outputs['modality_importance'],
            'predictions': torch.softmax(logits, dim=-1)
        }
    
    def predict_with_fallback(self, text_ids=None, images=None, audio_features=None, 
                            confidence_threshold=0.7):
        """Predict with automatic fallback to available modalities."""
        
        # Try with all available modalities
        result = self.forward(text_ids, images, audio_features)
        
        if result['confidence'].item() >= confidence_threshold:
            return result
        
        # Fallback strategies
        fallback_results = []
        
        # Try text + visual
        if text_ids is not None and images is not None:
            result_tv = self.forward(text_ids, images, None)
            fallback_results.append(('text+visual', result_tv))
        
        # Try text only
        if text_ids is not None:
            result_t = self.forward(text_ids, None, None)
            fallback_results.append(('text', result_t))
        
        # Try visual only
        if images is not None:
            result_v = self.forward(None, images, None)
            fallback_results.append(('visual', result_v))
        
        # Select best fallback
        if fallback_results:
            best_result = max(fallback_results, key=lambda x: x[1]['confidence'].item())
            return {**best_result[1], 'fallback_strategy': best_result[0]}
        
        return result  # Return original if no fallback available