class MetacognitiveTokenizer:
    def __init__(self, base_tokenizer):
        self.base_tokenizer = base_tokenizer
        self.meta_tokens = {
            'meta_start': '[META]',
            'meta_end': '[/META]',
            'strategy': '[STRATEGY]',
            'monitor': '[MONITOR]',
            'evaluate': '[EVALUATE]',
            'adapt': '[ADAPT]',
            'confidence': '[META_CONF]'
        }
        
    def encode_with_metacognition(self, problem):
        """Encode problem with metacognitive prompting"""
        prompt = (
            f"Problem: {problem}\n"
            f"{self.meta_tokens['meta_start']}\n"
            f"{self.meta_tokens['strategy']} Selecting problem-solving approach...\n"
        )
        return self.base_tokenizer.encode(prompt)
    
    def structure_metacognitive_reasoning(self, problem_type, reasoning_process):
        """Structure reasoning with metacognitive monitoring"""
        structured = []
        
        # Strategy selection
        structured.append(self.meta_tokens['strategy'])
        strategy = self._select_strategy(problem_type)
        structured.append(f"Selected strategy: {strategy}")
        
        # Monitor reasoning progress
        for i, step in enumerate(reasoning_process):
            structured.append(f"Step {i+1}: {step}")
            
            # Metacognitive monitoring
            structured.append(self.meta_tokens['monitor'])
            monitoring = self._monitor_progress(step, i, len(reasoning_process))
            structured.append(monitoring)
            
            # Evaluate if adaptation needed
            if self._needs_adaptation(monitoring):
                structured.append(self.meta_tokens['adapt'])
                adaptation = self._adapt_strategy(strategy, monitoring)
                structured.append(f"Adapting approach: {adaptation}")
        
        # Final evaluation
        structured.append(self.meta_tokens['evaluate'])
        evaluation = self._evaluate_solution(reasoning_process)
        structured.append(evaluation)
        
        # Confidence assessment
        structured.append(self.meta_tokens['confidence'])
        confidence = self._assess_confidence(evaluation)
        structured.append(f"Overall confidence: {confidence}")
        
        structured.append(self.meta_tokens['meta_end'])
        
        return structured
    
    def _select_strategy(self, problem_type):
        """Select appropriate problem-solving strategy"""
        strategies = {
            'mathematical': 'Step-by-step calculation with verification',
            'logical': 'Deductive reasoning with premise checking',
            'creative': 'Brainstorming with evaluation',
            'analytical': 'Breaking down into components'
        }
        return strategies.get(problem_type, 'General problem-solving')
    
    def _monitor_progress(self, step, current, total):
        """Monitor reasoning progress"""
        progress = (current + 1) / total
        if progress < 0.3:
            return "Early stage: Establishing foundation"
        elif progress < 0.7:
            return "Mid stage: Developing solution"
        else:
            return "Final stage: Converging on answer"
    
    def _needs_adaptation(self, monitoring):
        """Determine if strategy adaptation is needed"""
        adaptation_triggers = ['stuck', 'error', 'unclear', 'complex']
        return any(trigger in monitoring.lower() for trigger in adaptation_triggers)
    
    def _adapt_strategy(self, current_strategy, monitoring):
        """Adapt problem-solving strategy"""
        return f"Switching from {current_strategy} to more detailed analysis"
    
    def _evaluate_solution(self, reasoning_process):
        """Evaluate the quality of the solution"""
        return "Solution appears complete and logically sound"
    
    def _assess_confidence(self, evaluation):
        """Assess confidence in the solution"""
        if 'complete' in evaluation and 'sound' in evaluation:
            return "High (90%)"
        elif 'partial' in evaluation:
            return "Medium (60%)"
        else:
            return "Low (30%)"