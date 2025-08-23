"""
Attention pattern analysis with special tokens

Extracted from: part1/chapter01/role_in_attention.tex
Block: 1
Lines: 57
"""

import torch
import torch.nn.functional as F

def analyze_special_token_attention(attention_weights, token_ids, special_tokens):
    """
    Analyze attention patterns involving special tokens
    
    Args:
        attention_weights: [batch_size, num_heads, seq_len, seq_len]
        token_ids: [batch_size, seq_len] 
        special_tokens: dict mapping token names to ids
    """
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    
    # Find special token positions
    cls_positions = (token_ids == special_tokens['CLS']).nonzero()
    sep_positions = (token_ids == special_tokens['SEP']).nonzero()
    
    results = {}
    
    # Analyze CLS token attention patterns
    for batch_idx, pos_idx in cls_positions:
        cls_attention = attention_weights[batch_idx, :, pos_idx, :]
        
        # Average across heads for analysis
        avg_attention = cls_attention.mean(dim=0)
        
        # Compute attention entropy (measure of focus)
        attention_entropy = -torch.sum(avg_attention * torch.log(avg_attention + 1e-10))
        
        # Find top attended positions
        top_positions = torch.topk(avg_attention, k=5).indices
        
        results[f'CLS_batch_{batch_idx}'] = {
            'entropy': attention_entropy.item(),
            'top_positions': top_positions.tolist(),
            'attention_distribution': avg_attention
        }
    
    # Analyze cross-segment attention through SEP tokens
    if len(sep_positions) > 0:
        for batch_idx, sep_pos in sep_positions:
            # Attention from content tokens to SEP token
            to_sep = attention_weights[batch_idx, :, :, sep_pos].mean(dim=0)
            
            # Attention from SEP token to content tokens  
            from_sep = attention_weights[batch_idx, :, sep_pos, :].mean(dim=0)
            
            results[f'SEP_batch_{batch_idx}_pos_{sep_pos}'] = {
                'receives_attention': to_sep,
                'gives_attention': from_sep,
                'bidirectional_strength': torch.mean(to_sep + from_sep)
            }
    
    return results

# Example usage for attention pattern visualization
def visualize_special_token_attention(model, tokenizer, text):
    """Visualize attention patterns involving special tokens"""
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attention_weights = outputs.attentions[-1]  # Last layer attention
    
    special_tokens = {
        'CLS': tokenizer.cls_token_id,
        'SEP': tokenizer.sep_token_id,
        'PAD': tokenizer.pad_token_id
    }
    
    return analyze_special_token_attention(
        attention_weights, inputs['input_ids'], special_tokens
    )