class ThinkingTokenizer:
    def __init__(self, base_tokenizer):
        self.base_tokenizer = base_tokenizer
        self.thinking_tokens = {
            'think_start': '<think>',
            'think_end': '</think>',
            'output_start': '<output>',
            'output_end': '</output>'
        }
        
    def encode_with_thinking(self, prompt):
        """Encode prompt to encourage thinking before answering"""
        formatted_prompt = (
            f"{prompt}\n"
            f"{self.thinking_tokens['think_start']}\n"
            f"Let me work through this step by step...\n"
        )
        return self.base_tokenizer.encode(formatted_prompt)
    
    def separate_thinking_from_output(self, generated_text):
        """Extract thinking process and final output separately"""
        import re
        
        # Extract thinking section
        think_pattern = f"{re.escape(self.thinking_tokens['think_start'])}(.*?){re.escape(self.thinking_tokens['think_end'])}"
        thinking_match = re.search(think_pattern, generated_text, re.DOTALL)
        thinking = thinking_match.group(1).strip() if thinking_match else ""
        
        # Extract output section
        output_pattern = f"{re.escape(self.thinking_tokens['output_start'])}(.*?){re.escape(self.thinking_tokens['output_end'])}"
        output_match = re.search(output_pattern, generated_text, re.DOTALL)
        output = output_match.group(1).strip() if output_match else ""
        
        # If no output tags, everything after thinking is output
        if not output and thinking:
            remaining = generated_text.split(self.thinking_tokens['think_end'])[-1].strip()
            output = remaining
        
        return {
            'thinking': thinking,
            'output': output,
            'full_response': generated_text
        }
    
    def format_for_training(self, question, thinking_steps, answer):
        """Format training data with explicit thinking"""
        formatted = (
            f"Question: {question}\n"
            f"{self.thinking_tokens['think_start']}\n"
        )
        
        for i, step in enumerate(thinking_steps, 1):
            formatted += f"Step {i}: {step}\n"
        
        formatted += (
            f"{self.thinking_tokens['think_end']}\n"
            f"{self.thinking_tokens['output_start']}\n"
            f"{answer}\n"
            f"{self.thinking_tokens['output_end']}"
        )
        
        return formatted