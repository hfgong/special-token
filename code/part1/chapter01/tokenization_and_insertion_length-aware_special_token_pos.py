"""
Length-aware special token positioning

Extracted from: part1/chapter01/tokenization_and_insertion.tex
Block: 6
Lines: 52
"""

def optimize_token_positioning(texts, max_length, special_tokens):
    """Optimize special token positioning for variable-length inputs"""
    
    def calculate_information_density(tokens):
        """Estimate information density of token segments"""
        # Simple heuristic: shorter, less common tokens have higher density
        density_scores = []
        for token in tokens:
            freq = token_frequency.get(token, 1)  # From pre-computed statistics
            density = 1.0 / (len(token) * math.log(freq + 1))
            density_scores.append(density)
        return density_scores
    
    def intelligent_truncation(tokens, target_length, special_token_count):
        """Truncate tokens while preserving high-information segments"""
        if len(tokens) <= target_length - special_token_count:
            return tokens
        
        densities = calculate_information_density(tokens)
        
        # Create segments and compute average density
        segment_size = 50  # Adjust based on typical sentence length
        segments = []
        for i in range(0, len(tokens), segment_size):
            segment_tokens = tokens[i:i + segment_size]
            segment_densities = densities[i:i + segment_size]
            avg_density = sum(segment_densities) / len(segment_densities)
            segments.append({
                'tokens': segment_tokens,
                'start': i,
                'density': avg_density
            })
        
        # Sort by density and keep highest-density segments
        segments.sort(key=lambda x: x['density'], reverse=True)
        
        selected_tokens = []
        remaining_length = target_length - special_token_count
        
        for segment in segments:
            if len(selected_tokens) + len(segment['tokens']) <= remaining_length:
                selected_tokens.extend(segment['tokens'])
            else:
                # Partial segment inclusion
                remaining_space = remaining_length - len(selected_tokens)
                selected_tokens.extend(segment['tokens'][:remaining_space])
                break
        
        return selected_tokens
    
    optimized_sequences = []
    for text in texts:
        tokens = tokenize(text)  # Basic tokenization
        
        # Apply intelligent truncation
        optimal_tokens = intelligent_truncation(
            tokens, max_length, len(special_tokens)
        )
        
        # Insert special tokens
        final_sequence = insert_special_tokens(optimal_tokens, special_tokens)
        
        optimized_sequences.append(final_sequence)
    
    return optimized_sequences