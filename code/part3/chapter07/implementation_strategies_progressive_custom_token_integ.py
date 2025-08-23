"""
Progressive custom token integration

Extracted from: part3/chapter07/implementation_strategies.tex
Block: 2
Lines: 188
"""

class ProgressiveTokenIntegrator:
    def __init__(self, base_model, custom_tokens):
        self.base_model = base_model
        self.custom_tokens = custom_tokens
        self.integration_schedule = self.create_integration_schedule()
        
    def create_integration_schedule(self):
        """Create schedule for progressive token integration."""
        schedule = []
        
        # Sort tokens by complexity and dependencies
        sorted_tokens = self.sort_tokens_by_complexity()
        
        for phase, token_group in enumerate(sorted_tokens):
            schedule.append({
                'phase': phase,
                'tokens': token_group,
                'warmup_steps': 1000 * (phase + 1),
                'learning_rate_multiplier': 0.1 * (phase + 1),
                'stability_checks': self.get_stability_checks(token_group)
            })
        
        return schedule
    
    def integrate_token_group(self, token_group, phase_config):
        """Integrate a group of tokens according to phase configuration."""
        # Add tokens to model
        for token in token_group:
            self.add_token_to_model(token)
        
        # Configure learning rates
        optimizer_config = self.create_phase_optimizer_config(phase_config)
        
        # Training loop with stability monitoring
        for step in range(phase_config['warmup_steps']):
            # Training step
            loss = self.training_step(optimizer_config)
            
            # Stability monitoring
            if step % 100 == 0:
                stability_results = self.check_stability(token_group)
                if not stability_results['stable']:
                    self.apply_stability_corrections(token_group, stability_results)
            
            # Learning rate adjustment
            if step % 500 == 0:
                self.adjust_learning_rates(token_group, loss)
    
    def check_stability(self, token_group):
        """Check training stability for token group."""
        stability_checks = {}
        
        for token in token_group:
            token_stability = {}
            
            # Check embedding gradient norms
            embedding_grad = token.embedding.grad
            if embedding_grad is not None:
                grad_norm = torch.norm(embedding_grad)
                token_stability['grad_norm'] = grad_norm
                token_stability['grad_stable'] = grad_norm < 10.0
            
            # Check attention pattern stability
            attention_patterns = self.extract_token_attention_patterns(token)
            token_stability['attention_entropy'] = self.compute_attention_entropy(
                attention_patterns
            )
            token_stability['attention_stable'] = (
                token_stability['attention_entropy'] > 1.0
            )
            
            # Check output contribution stability
            output_contribution = self.measure_token_output_contribution(token)
            token_stability['contribution_magnitude'] = output_contribution
            token_stability['contribution_stable'] = (
                0.01 < output_contribution < 0.5
            )
            
            stability_checks[token.name] = token_stability
        
        # Overall stability assessment
        overall_stable = all(
            check['grad_stable'] and check['attention_stable'] and check['contribution_stable']
            for check in stability_checks.values()
        )
        
        return {
            'stable': overall_stable,
            'token_details': stability_checks,
            'recommendations': self.generate_stability_recommendations(stability_checks)
        }
    
    def apply_stability_corrections(self, token_group, stability_results):
        """Apply corrections based on stability analysis."""
        for token in token_group:
            token_stability = stability_results['token_details'][token.name]
            
            if not token_stability['grad_stable']:
                # Apply gradient clipping
                self.apply_gradient_clipping(token, max_norm=1.0)
            
            if not token_stability['attention_stable']:
                # Adjust attention temperature
                self.adjust_attention_temperature(token, factor=1.1)
            
            if not token_stability['contribution_stable']:
                # Scale learning rate
                contribution = token_stability['contribution_magnitude']
                if contribution > 0.5:
                    self.scale_token_learning_rate(token, factor=0.5)
                elif contribution < 0.01:
                    self.scale_token_learning_rate(token, factor=2.0)

class CustomTokenTrainer:
    def __init__(self, base_model, custom_tokens, training_config):
        self.base_model = base_model
        self.custom_tokens = custom_tokens
        self.training_config = training_config
        
        # Initialize training components
        self.setup_optimizers()
        self.setup_schedulers()
        self.setup_monitoring()
    
    def setup_optimizers(self):
        """Setup separate optimizers for custom tokens."""
        self.optimizers = {}
        
        # Base model optimizer
        base_params = [
            p for p in self.base_model.parameters() 
            if not any(p is token.embedding for token in self.custom_tokens)
        ]
        self.optimizers['base'] = torch.optim.AdamW(
            base_params, 
            lr=self.training_config['base_lr'],
            weight_decay=self.training_config['weight_decay']
        )
        
        # Custom token optimizers
        for token in self.custom_tokens:
            self.optimizers[token.name] = torch.optim.AdamW(
                [token.embedding],
                lr=self.training_config['token_lr'],
                weight_decay=self.training_config['token_weight_decay']
            )
    
    def setup_schedulers(self):
        """Setup learning rate schedulers."""
        self.schedulers = {}
        
        for name, optimizer in self.optimizers.items():
            if name == 'base':
                self.schedulers[name] = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=self.training_config['total_steps']
                )
            else:
                # Custom warmup schedule for tokens
                self.schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=self.create_token_lr_schedule()
                )
    
    def create_token_lr_schedule(self):
        """Create learning rate schedule for custom tokens."""
        def lr_lambda(step):
            warmup_steps = self.training_config['token_warmup_steps']
            if step < warmup_steps:
                return step / warmup_steps
            else:
                remaining_steps = self.training_config['total_steps'] - warmup_steps
                progress = (step - warmup_steps) / remaining_steps
                return 0.5 * (1 + torch.cos(torch.pi * progress))
        
        return lr_lambda
    
    def training_step(self, batch):
        """Perform single training step with custom token considerations."""
        # Forward pass
        outputs = self.base_model(batch['input_ids'])
        loss = self.compute_loss(outputs, batch)
        
        # Add custom token regularization
        token_regularization = self.compute_token_regularization()
        total_loss = loss + token_regularization
        
        # Backward pass
        total_loss.backward()
        
        # Apply custom token specific gradient processing
        self.process_custom_token_gradients()
        
        # Optimizer steps
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad()
        
        # Scheduler steps
        for scheduler in self.schedulers.values():
            scheduler.step()
        
        return {
            'loss': loss.item(),
            'token_regularization': token_regularization.item(),
            'total_loss': total_loss.item()
        }
    
    def compute_token_regularization(self):
        """Compute regularization terms for custom tokens."""
        regularization = torch.tensor(0.0, device=self.base_model.device)
        
        for token in self.custom_tokens:
            # Embedding norm regularization
            norm_penalty = torch.norm(token.embedding) ** 2
            regularization += self.training_config['norm_penalty_weight'] * norm_penalty
            
            # Similarity penalty (prevent tokens from becoming too similar)
            for other_token in self.custom_tokens:
                if token != other_token:
                    similarity = torch.cosine_similarity(
                        token.embedding.unsqueeze(0),
                        other_token.embedding.unsqueeze(0),
                        dim=1
                    )
                    similarity_penalty = torch.relu(similarity - 0.8) ** 2
                    regularization += self.training_config['similarity_penalty_weight'] * similarity_penalty
        
        return regularization