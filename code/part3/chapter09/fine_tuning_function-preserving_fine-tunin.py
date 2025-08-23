"""
Function-preserving fine-tuning framework

Extracted from: part3/chapter09/fine_tuning.tex
Block: 1
Lines: 364
"""

class FunctionPreservingFineTuner:
    def __init__(self, pretrained_model, special_tokens, fine_tuning_config):
        self.pretrained_model = pretrained_model
        self.special_tokens = special_tokens
        self.config = fine_tuning_config
        
        # Fine-tuning components
        self.parameter_selector = ParameterSelector()
        self.function_monitor = SpecialTokenFunctionMonitor()
        self.adaptation_controller = AdaptationController()
        
        # Fine-tuning state
        self.baseline_functions = None
        self.fine_tuning_history = []
        
    def execute_function_preserving_fine_tuning(self, downstream_data, task_config):
        """Execute fine-tuning while preserving special token functions."""
        fine_tuning_results = {}
        
        # Establish baseline function measurements
        self.baseline_functions = self.measure_baseline_functions()
        
        # Design fine-tuning strategy
        fine_tuning_strategy = self.design_fine_tuning_strategy(task_config)
        
        # Execute fine-tuning phases
        for phase_idx, phase_config in enumerate(fine_tuning_strategy['phases']):
            phase_results = self.execute_fine_tuning_phase(
                phase_config, downstream_data, task_config
            )
            
            fine_tuning_results[f'phase_{phase_idx}'] = phase_results
            
            # Monitor function preservation
            function_status = self.monitor_function_preservation(phase_results)
            
            # Apply corrective measures if needed
            if function_status['requires_correction']:
                correction_results = self.apply_function_corrections(
                    function_status, phase_config
                )
                fine_tuning_results[f'phase_{phase_idx}_corrections'] = correction_results
        
        # Final validation
        final_validation = self.validate_fine_tuning_results(fine_tuning_results)
        fine_tuning_results['final_validation'] = final_validation
        
        return fine_tuning_results
    
    def measure_baseline_functions(self):
        """Measure baseline special token functions before fine-tuning."""
        baseline_measurements = {}
        
        for token_name in self.special_tokens:
            token_functions = self.function_monitor.measure_token_functions(
                self.pretrained_model, token_name
            )
            baseline_measurements[token_name] = token_functions
        
        return baseline_measurements
    
    def design_fine_tuning_strategy(self, task_config):
        """Design fine-tuning strategy based on task requirements."""
        strategy = {
            'phases': [],
            'parameter_groups': self.identify_parameter_groups(),
            'learning_rates': self.compute_phase_learning_rates(task_config),
            'regularization': self.design_regularization_strategy(task_config)
        }
        
        # Phase 1: Minimal adaptation
        strategy['phases'].append({
            'name': 'minimal_adaptation',
            'duration_epochs': self.config['minimal_adaptation_epochs'],
            'parameter_groups': ['task_head', 'top_layers'],
            'special_token_adaptation': 'frozen',
            'learning_rate_multiplier': 0.1
        })
        
        # Phase 2: Gradual adaptation
        strategy['phases'].append({
            'name': 'gradual_adaptation',
            'duration_epochs': self.config['gradual_adaptation_epochs'],
            'parameter_groups': ['task_head', 'top_layers', 'middle_layers'],
            'special_token_adaptation': 'constrained',
            'learning_rate_multiplier': 0.5
        })
        
        # Phase 3: Full adaptation (if needed)
        if task_config.get('requires_full_adaptation', False):
            strategy['phases'].append({
                'name': 'full_adaptation',
                'duration_epochs': self.config['full_adaptation_epochs'],
                'parameter_groups': 'all',
                'special_token_adaptation': 'regularized',
                'learning_rate_multiplier': 1.0
            })
        
        return strategy
    
    def execute_fine_tuning_phase(self, phase_config, downstream_data, task_config):
        """Execute single fine-tuning phase."""
        phase_results = {
            'phase_name': phase_config['name'],
            'training_metrics': {},
            'function_preservation_metrics': {},
            'task_performance_metrics': {}
        }
        
        # Configure optimizer for phase
        optimizer = self.configure_phase_optimizer(phase_config)
        
        # Configure special token handling
        special_token_handler = self.configure_special_token_handling(phase_config)
        
        # Execute training epochs
        for epoch in range(phase_config['duration_epochs']):
            epoch_results = self.execute_fine_tuning_epoch(
                epoch, downstream_data, optimizer, special_token_handler, task_config
            )
            
            # Record metrics
            phase_results['training_metrics'][f'epoch_{epoch}'] = epoch_results['training_metrics']
            
            # Monitor function preservation
            if epoch % self.config['function_monitoring_frequency'] == 0:
                function_metrics = self.monitor_function_preservation_during_training()
                phase_results['function_preservation_metrics'][f'epoch_{epoch}'] = function_metrics
            
            # Evaluate task performance
            if epoch % self.config['task_evaluation_frequency'] == 0:
                task_metrics = self.evaluate_task_performance(downstream_data, task_config)
                phase_results['task_performance_metrics'][f'epoch_{epoch}'] = task_metrics
        
        return phase_results
    
    def configure_special_token_handling(self, phase_config):
        """Configure special token handling for current phase."""
        adaptation_mode = phase_config['special_token_adaptation']
        
        if adaptation_mode == 'frozen':
            return FrozenSpecialTokenHandler(self.special_tokens)
        elif adaptation_mode == 'constrained':
            return ConstrainedSpecialTokenHandler(
                self.special_tokens, 
                self.baseline_functions,
                self.config['constraint_strength']
            )
        elif adaptation_mode == 'regularized':
            return RegularizedSpecialTokenHandler(
                self.special_tokens,
                self.baseline_functions,
                self.config['regularization_strength']
            )
        else:
            return StandardSpecialTokenHandler(self.special_tokens)
    
    def monitor_function_preservation(self, phase_results):
        """Monitor preservation of special token functions."""
        current_functions = {}
        
        for token_name in self.special_tokens:
            current_functions[token_name] = self.function_monitor.measure_token_functions(
                self.pretrained_model, token_name
            )
        
        # Compare with baseline
        preservation_status = {}
        overall_preservation_quality = 0.0
        
        for token_name, current_func in current_functions.items():
            baseline_func = self.baseline_functions[token_name]
            
            preservation_metrics = self.compute_preservation_metrics(
                baseline_func, current_func
            )
            
            preservation_status[token_name] = preservation_metrics
            overall_preservation_quality += preservation_metrics['preservation_score']
        
        overall_preservation_quality /= len(self.special_tokens)
        
        return {
            'overall_preservation_quality': overall_preservation_quality,
            'token_specific_preservation': preservation_status,
            'requires_correction': overall_preservation_quality < self.config['min_preservation_threshold']
        }
    
    def compute_preservation_metrics(self, baseline_func, current_func):
        """Compute function preservation metrics."""
        metrics = {}
        
        # Functional similarity
        metrics['functional_similarity'] = self.compute_functional_similarity(
            baseline_func, current_func
        )
        
        # Representation quality
        metrics['representation_quality'] = self.compute_representation_quality(
            baseline_func, current_func
        )
        
        # Attention pattern preservation
        metrics['attention_pattern_preservation'] = self.compute_attention_pattern_preservation(
            baseline_func, current_func
        )
        
        # Overall preservation score
        metrics['preservation_score'] = (
            0.4 * metrics['functional_similarity'] +
            0.3 * metrics['representation_quality'] +
            0.3 * metrics['attention_pattern_preservation']
        )
        
        return metrics

class ConstrainedSpecialTokenHandler:
    def __init__(self, special_tokens, baseline_functions, constraint_strength):
        self.special_tokens = special_tokens
        self.baseline_functions = baseline_functions
        self.constraint_strength = constraint_strength
        
    def apply_constraints(self, model, loss, current_step):
        """Apply constraints to preserve special token functions."""
        constraint_loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        
        for token_name in self.special_tokens:
            # Measure current function deviation
            current_functions = self.measure_current_functions(model, token_name)
            baseline_functions = self.baseline_functions[token_name]
            
            # Compute constraint violations
            violations = self.compute_constraint_violations(
                baseline_functions, current_functions
            )
            
            # Add constraint penalty
            constraint_penalty = self.compute_constraint_penalty(violations)
            constraint_loss = constraint_loss + self.constraint_strength * constraint_penalty
        
        return loss + constraint_loss
    
    def compute_constraint_violations(self, baseline_functions, current_functions):
        """Compute constraint violations for special token functions."""
        violations = {}
        
        # Embedding norm violations
        baseline_norm = baseline_functions.get('embedding_norm', 1.0)
        current_norm = current_functions.get('embedding_norm', 1.0)
        violations['embedding_norm'] = torch.relu(torch.abs(current_norm - baseline_norm) - 0.1)
        
        # Attention pattern violations
        baseline_patterns = baseline_functions.get('attention_patterns')
        current_patterns = current_functions.get('attention_patterns')
        if baseline_patterns is not None and current_patterns is not None:
            pattern_similarity = torch.cosine_similarity(
                baseline_patterns.flatten(), current_patterns.flatten(), dim=0
            )
            violations['attention_patterns'] = torch.relu(0.8 - pattern_similarity)
        
        # Functional output violations
        baseline_outputs = baseline_functions.get('functional_outputs')
        current_outputs = current_functions.get('functional_outputs')
        if baseline_outputs is not None and current_outputs is not None:
            output_similarity = torch.cosine_similarity(
                baseline_outputs.flatten(), current_outputs.flatten(), dim=0
            )
            violations['functional_outputs'] = torch.relu(0.7 - output_similarity)
        
        return violations
    
    def compute_constraint_penalty(self, violations):
        """Compute penalty for constraint violations."""
        total_penalty = torch.tensor(0.0, requires_grad=True)
        
        for violation_type, violation_magnitude in violations.items():
            # Apply different penalty weights for different violation types
            if violation_type == 'embedding_norm':
                penalty_weight = 1.0
            elif violation_type == 'attention_patterns':
                penalty_weight = 2.0
            elif violation_type == 'functional_outputs':
                penalty_weight = 3.0
            else:
                penalty_weight = 1.0
            
            total_penalty = total_penalty + penalty_weight * violation_magnitude.pow(2)
        
        return total_penalty

class TaskAdaptiveFineTuner:
    def __init__(self, model, special_tokens):
        self.model = model
        self.special_tokens = special_tokens
        
        # Task adaptation components
        self.task_analyzer = TaskAnalyzer()
        self.adaptation_strategy_selector = AdaptationStrategySelector()
        self.performance_optimizer = PerformanceOptimizer()
        
    def task_adaptive_fine_tuning(self, downstream_task, training_data):
        """Adapt fine-tuning strategy based on task characteristics."""
        adaptation_results = {}
        
        # Analyze task characteristics
        task_analysis = self.task_analyzer.analyze_task(downstream_task, training_data)
        
        # Select appropriate adaptation strategy
        adaptation_strategy = self.adaptation_strategy_selector.select_strategy(
            task_analysis, self.special_tokens
        )
        
        # Execute adaptive fine-tuning
        for strategy_phase in adaptation_strategy['phases']:
            phase_results = self.execute_adaptive_phase(
                strategy_phase, training_data, task_analysis
            )
            adaptation_results[strategy_phase['name']] = phase_results
        
        return adaptation_results
    
    def execute_adaptive_phase(self, strategy_phase, training_data, task_analysis):
        """Execute adaptive fine-tuning phase."""
        phase_results = {}
        
        # Configure phase-specific adaptations
        if strategy_phase['type'] == 'special_token_specialization':
            phase_results = self.execute_specialization_phase(
                strategy_phase, training_data, task_analysis
            )
        elif strategy_phase['type'] == 'attention_adaptation':
            phase_results = self.execute_attention_adaptation_phase(
                strategy_phase, training_data, task_analysis
            )
        elif strategy_phase['type'] == 'representation_alignment':
            phase_results = self.execute_alignment_phase(
                strategy_phase, training_data, task_analysis
            )
        
        return phase_results
    
    def execute_specialization_phase(self, strategy_phase, training_data, task_analysis):
        """Execute special token specialization for task requirements."""
        specialization_results = {}
        
        # Identify specialization targets
        specialization_targets = strategy_phase['specialization_targets']
        
        for target in specialization_targets:
            token_name = target['token']
            specialization_type = target['specialization']
            
            if specialization_type == 'task_specific_aggregation':
                result = self.specialize_for_task_aggregation(
                    token_name, training_data, task_analysis
                )
            elif specialization_type == 'domain_adaptation':
                result = self.specialize_for_domain_adaptation(
                    token_name, training_data, task_analysis
                )
            elif specialization_type == 'performance_optimization':
                result = self.specialize_for_performance_optimization(
                    token_name, training_data, task_analysis
                )
            
            specialization_results[f'{token_name}_{specialization_type}'] = result
        
        return specialization_results
    
    def specialize_for_task_aggregation(self, token_name, training_data, task_analysis):
        """Specialize token for task-specific aggregation requirements."""
        aggregation_config = {
            'aggregation_type': task_analysis['aggregation_requirements'],
            'information_density': task_analysis['information_density'],
            'sequence_characteristics': task_analysis['sequence_characteristics']
        }
        
        # Create task-specific aggregation objective
        aggregation_objective = TaskSpecificAggregationObjective(
            token_name, aggregation_config
        )
        
        # Fine-tune with aggregation objective
        specialization_optimizer = torch.optim.AdamW(
            [param for name, param in self.model.named_parameters() 
             if token_name in name or 'attention' in name],
            lr=1e-5
        )
        
        for epoch in range(self.config['specialization_epochs']):
            for batch in training_data:
                specialization_optimizer.zero_grad()
                
                outputs = self.model(batch['input_ids'])
                
                # Compute specialization loss
                specialization_loss = aggregation_objective.compute_loss(outputs, batch)
                
                specialization_loss.backward()
                specialization_optimizer.step()
        
        return {
            'specialization_type': 'task_specific_aggregation',
            'final_specialization_quality': self.measure_aggregation_quality(token_name),
            'convergence_steps': epoch * len(training_data)
        }

class RegularizedSpecialTokenHandler:
    def __init__(self, special_tokens, baseline_functions, regularization_strength):
        self.special_tokens = special_tokens
        self.baseline_functions = baseline_functions
        self.regularization_strength = regularization_strength
        
    def apply_regularization(self, model, loss):
        """Apply regularization to preserve special token functions."""
        regularization_loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        
        for token_name in self.special_tokens:
            # Function preservation regularization
            function_reg = self.compute_function_preservation_regularization(
                model, token_name
            )
            
            # Embedding stability regularization
            embedding_reg = self.compute_embedding_stability_regularization(
                model, token_name
            )
            
            # Attention pattern regularization
            attention_reg = self.compute_attention_pattern_regularization(
                model, token_name
            )
            
            token_regularization = function_reg + embedding_reg + attention_reg
            regularization_loss = regularization_loss + token_regularization
        
        total_loss = loss + self.regularization_strength * regularization_loss
        return total_loss
    
    def compute_function_preservation_regularization(self, model, token_name):
        """Compute regularization for function preservation."""
        current_embedding = self.get_token_embedding(model, token_name)
        baseline_embedding = self.baseline_functions[token_name]['embedding']
        
        # L2 distance from baseline
        embedding_distance = torch.norm(current_embedding - baseline_embedding, p=2)
        
        # Cosine similarity preservation
        cosine_similarity = torch.cosine_similarity(
            current_embedding.unsqueeze(0), 
            baseline_embedding.unsqueeze(0),
            dim=1
        )
        similarity_loss = torch.relu(0.9 - cosine_similarity)
        
        function_regularization = embedding_distance + similarity_loss
        return function_regularization