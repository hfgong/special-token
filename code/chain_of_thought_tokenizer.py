class ChainOfThoughtTokenizer:
    def __init__(self, base_tokenizer):
        self.base_tokenizer = base_tokenizer
        self.cot_tokens = {
            'chain_start': '[COT_START]',
            'chain_end': '[COT_END]',
            'step': '[STEP]',
            'substep': '[SUBSTEP]',
            'reasoning': '[REASONING]',
            'calculation': '[CALC]',
            'verification': '[VERIFY]',
            'conclusion': '[CONCLUDE]'
        }
        
    def encode_reasoning_chain(self, problem, max_steps=10):
        """Encode problem with chain-of-thought prompting"""
        prompt = (
            f"Problem: {problem}\n"
            f"{self.cot_tokens['chain_start']}\n"
            f"I'll solve this step by step.\n"
        )
        
        # Add step tokens to guide reasoning
        for i in range(1, min(max_steps + 1, 5)):
            prompt += f"{self.cot_tokens['step']} {i}: \n"
        
        return self.base_tokenizer.encode(prompt)
    
    def structure_reasoning_steps(self, steps):
        """Structure reasoning steps with appropriate tokens"""
        structured = [self.cot_tokens['chain_start']]
        
        for i, step in enumerate(steps, 1):
            # Main step marker
            structured.append(f"{self.cot_tokens['step']} {i}:")
            
            # Classify step type and add appropriate token
            if self._is_calculation(step):
                structured.append(self.cot_tokens['calculation'])
            elif self._is_verification(step):
                structured.append(self.cot_tokens['verification'])
            else:
                structured.append(self.cot_tokens['reasoning'])
            
            structured.append(step)
            
            # Add substeps if the step is complex
            substeps = self._extract_substeps(step)
            for substep in substeps:
                structured.append(self.cot_tokens['substep'])
                structured.append(substep)
        
        # Add conclusion
        structured.append(self.cot_tokens['conclusion'])
        structured.append(self.cot_tokens['chain_end'])
        
        return structured
    
    def _is_calculation(self, step):
        """Determine if step involves calculation"""
        calc_keywords = ['calculate', 'compute', 'equals', '=', '+', '-', '*', '/']
        return any(keyword in step.lower() for keyword in calc_keywords)
    
    def _is_verification(self, step):
        """Determine if step involves verification"""
        verify_keywords = ['check', 'verify', 'confirm', 'validate', 'ensure']
        return any(keyword in step.lower() for keyword in verify_keywords)
    
    def _extract_substeps(self, step):
        """Extract substeps from a complex step"""
        # Simple heuristic: split on certain markers
        markers = ['First,', 'Second,', 'Then,', 'Next,', 'Finally,']
        substeps = []
        
        for marker in markers:
            if marker in step:
                parts = step.split(marker)
                if len(parts) > 1:
                    substeps.extend([part.strip() for part in parts[1:] if part.strip()])
        
        return substeps