class ConstitutionalReasoningTokenizer:
    def __init__(self, base_tokenizer):
        self.base_tokenizer = base_tokenizer
        self.principle_tokens = {
            'helpful': '[HELPFUL]',
            'harmless': '[HARMLESS]',
            'honest': '[HONEST]',
            'principle_check': '[PRINCIPLE_CHECK]',
            'revision_needed': '[REVISE_FOR_PRINCIPLE]',
            'approved': '[PRINCIPLE_APPROVED]'
        }
        self.principles = {
            'helpful': "Provide useful, relevant, and constructive assistance",
            'harmless': "Avoid generating harmful, dangerous, or inappropriate content",
            'honest': "Be truthful and acknowledge limitations and uncertainties"
        }
        
    def encode_with_principles(self, prompt):
        """Encode prompt with constitutional principles"""
        formatted = (
            f"{prompt}\n"
            f"{self.principle_tokens['principle_check']}\n"
            f"Checking response against principles:\n"
            f"{self.principle_tokens['helpful']} - {self.principles['helpful']}\n"
            f"{self.principle_tokens['harmless']} - {self.principles['harmless']}\n"
            f"{self.principle_tokens['honest']} - {self.principles['honest']}\n"
        )
        return self.base_tokenizer.encode(formatted)
    
    def check_response_principles(self, response):
        """Check if response adheres to constitutional principles"""
        checks = {}
        
        # Check helpfulness
        checks['helpful'] = self._check_helpfulness(response)
        
        # Check harmlessness
        checks['harmless'] = self._check_harmlessness(response)
        
        # Check honesty
        checks['honest'] = self._check_honesty(response)
        
        return checks
    
    def revise_for_principles(self, original_response, principle_violations):
        """Revise response to better adhere to principles"""
        revision_prompt = [self.principle_tokens['revision_needed']]
        
        for principle, violation in principle_violations.items():
            if violation:
                revision_prompt.append(
                    f"Revising for {principle}: {self.principles[principle]}"
                )
        
        # Add revision logic here
        revised_response = self._apply_principle_revision(
            original_response, 
            principle_violations
        )
        
        revision_prompt.append(self.principle_tokens['approved'])
        revision_prompt.append(revised_response)
        
        return revision_prompt
    
    def _check_helpfulness(self, response):
        """Check if response is helpful"""
        helpful_indicators = ['help', 'assist', 'solution', 'answer', 'explain']
        return any(indicator in response.lower() for indicator in helpful_indicators)
    
    def _check_harmlessness(self, response):
        """Check if response is harmless"""
        harmful_indicators = ['dangerous', 'illegal', 'harmful', 'malicious']
        return not any(indicator in response.lower() for indicator in harmful_indicators)
    
    def _check_honesty(self, response):
        """Check if response acknowledges uncertainty when appropriate"""
        uncertainty_indicators = ['not sure', 'might be', 'possibly', 'uncertain']
        absolute_claims = ['definitely', 'certainly', 'absolutely', 'guaranteed']
        
        # Simple heuristic: look for balance
        has_uncertainty = any(ind in response.lower() for ind in uncertainty_indicators)
        has_absolute = any(claim in response.lower() for claim in absolute_claims)
        
        return has_uncertainty or not has_absolute
    
    def _apply_principle_revision(self, response, violations):
        """Apply principle-based revisions to response"""
        revised = response
        
        if violations.get('helpful'):
            revised = f"Let me provide a more helpful response: {revised}"
        
        if violations.get('harmless'):
            revised = "[Content revised for safety]"
        
        if violations.get('honest'):
            revised = f"I should clarify that {revised}"
        
        return revised