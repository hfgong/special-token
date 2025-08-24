def analyze_token_interactions(model, dataloader, device):
    """Analyze interaction patterns between different token types."""
    model.eval()
    
    interactions = {
        'cls_to_register': [],
        'register_to_cls': [],
        'register_to_register': [],
        'register_to_patch': []
    }
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            # Forward pass with attention output
            output = model(batch, output_attentions=True)
            
            for layer_attention in output.attentions:
                # Average across batch and heads
                attention = layer_attention.mean(dim=(0, 1))  # [seq_len, seq_len]
                
                num_registers = model.num_register_tokens
                cls_idx = 0
                reg_start = 1
                reg_end = reg_start + num_registers
                patch_start = reg_end
                
                # Extract different interaction types
                cls_to_reg = attention[cls_idx, reg_start:reg_end].mean().item()
                reg_to_cls = attention[reg_start:reg_end, cls_idx].mean().item()
                
                reg_to_reg = attention[reg_start:reg_end, reg_start:reg_end]
                reg_to_reg_score = (reg_to_reg.sum() - reg_to_reg.diag().sum()) / (num_registers * (num_registers - 1))
                
                reg_to_patch = attention[reg_start:reg_end, patch_start:].mean().item()
                
                interactions['cls_to_register'].append(cls_to_reg)
                interactions['register_to_cls'].append(reg_to_cls)
                interactions['register_to_register'].append(reg_to_reg_score.item())
                interactions['register_to_patch'].append(reg_to_patch)
            
            # Limit analysis for efficiency
            if len(interactions['cls_to_register']) >= 500:
                break
    
    # Compute statistics
    results = {}
    for key, values in interactions.items():
        results[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values)
        }
    
    return results