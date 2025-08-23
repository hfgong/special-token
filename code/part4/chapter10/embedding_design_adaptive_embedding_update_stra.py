"""
Adaptive embedding update strategies

Extracted from: part4/chapter10/embedding_design.tex
Block: 2
Lines: 75
"""

class AdaptiveEmbeddingUpdater:
    def __init__(self, model, special_token_ids):
        self.model = model
        self.special_token_ids = set(special_token_ids)
        self.update_statistics = {}
        
    def create_adaptive_optimizer(self, base_lr=5e-5):
        """Create optimizer with different learning rates for special tokens."""
        
        # Separate parameters
        special_token_params = []
        regular_params = []
        
        for name, param in self.model.named_parameters():
            if 'embeddings.word_embeddings' in name:
                # Check if this embedding corresponds to special tokens
                if self._is_special_token_param(param):
                    special_token_params.append(param)
                else:
                    regular_params.append(param)
            else:
                regular_params.append(param)
                
        # Create optimizer with different learning rates
        optimizer = torch.optim.AdamW([
            {'params': regular_params, 'lr': base_lr},
            {'params': special_token_params, 'lr': base_lr * 2.0}  # Higher LR for special tokens
        ])
        
        return optimizer
        
    def apply_gradient_scaling(self, model):
        """Apply gradient scaling to special token embeddings."""
        embeddings = model.embeddings.word_embeddings
        
        # Register gradient hook
        def scale_gradients(grad):
            # Create scaling mask
            scaling_mask = torch.ones_like(grad)
            
            for token_id in self.special_token_ids:
                # Scale gradients for special tokens
                scaling_mask[token_id] *= 1.5  # Increase gradient magnitude
                
            return grad * scaling_mask
            
        embeddings.weight.register_hook(scale_gradients)
        
    def update_with_momentum(self, token_id, gradient, momentum=0.9):
        """Update special token embedding with momentum."""
        if token_id not in self.update_statistics:
            self.update_statistics[token_id] = {
                'momentum': torch.zeros_like(gradient),
                'update_count': 0
            }
            
        stats = self.update_statistics[token_id]
        
        # Update momentum
        stats['momentum'] = momentum * stats['momentum'] + (1 - momentum) * gradient
        stats['update_count'] += 1
        
        # Apply bias correction
        bias_correction = 1 - momentum ** stats['update_count']
        corrected_momentum = stats['momentum'] / bias_correction
        
        return corrected_momentum
        
    def adaptive_clipping(self, token_id, gradient, clip_value=1.0):
        """Apply adaptive gradient clipping for special tokens."""
        if token_id not in self.update_statistics:
            self.update_statistics[token_id] = {
                'grad_norm_history': [],
                'clip_value': clip_value
            }
            
        stats = self.update_statistics[token_id]
        
        # Track gradient norm
        grad_norm = gradient.norm().item()
        stats['grad_norm_history'].append(grad_norm)
        
        # Adapt clipping value based on history
        if len(stats['grad_norm_history']) > 100:
            # Use exponential moving average of gradient norms
            avg_norm = np.mean(stats['grad_norm_history'][-100:])
            std_norm = np.std(stats['grad_norm_history'][-100:])
            
            # Adaptive clipping threshold
            adaptive_clip = avg_norm + 2 * std_norm
            stats['clip_value'] = min(clip_value, adaptive_clip)
            
        # Apply clipping
        if grad_norm > stats['clip_value']:
            gradient = gradient * (stats['clip_value'] / grad_norm)
            
        return gradient