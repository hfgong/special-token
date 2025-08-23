"""
Audio-text retrieval evaluation

Extracted from: part2/chapter05/audio_tokens.tex
Block: 7
Lines: 53
"""

def evaluate_audio_text_retrieval(model, dataloader, device):
    """Evaluate audio-text retrieval performance."""
    
    model.eval()
    
    all_audio_features = []
    all_text_features = []
    
    with torch.no_grad():
        for batch in dataloader:
            audio_features = batch['audio_features'].to(device)
            text_ids = batch['text_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Extract features through multimodal model
            outputs = model(
                text_ids=text_ids,
                audio_features=audio_features,
                attention_mask=attention_mask
            )
            
            # Extract modality-specific representations
            hidden_states = outputs['hidden_states']
            modality_labels = outputs['modality_labels']
            
            # Pool audio and text features
            audio_mask = (modality_labels == 1)
            text_mask = (modality_labels == 0)
            
            audio_pooled = hidden_states[audio_mask.unsqueeze(-1).expand_as(hidden_states)].mean()
            text_pooled = hidden_states[text_mask.unsqueeze(-1).expand_as(hidden_states)].mean()
            
            all_audio_features.append(audio_pooled)
            all_text_features.append(text_pooled)
    
    # Compute retrieval metrics
    audio_features = torch.stack(all_audio_features)
    text_features = torch.stack(all_text_features)
    
    similarity_matrix = torch.matmul(audio_features, text_features.t())
    
    # Audio-to-text retrieval
    a2t_ranks = []
    for i in range(len(audio_features)):
        similarities = similarity_matrix[i]
        rank = (similarities >= similarities[i]).sum().item()
        a2t_ranks.append(rank)
    
    # Text-to-audio retrieval
    t2a_ranks = []
    for i in range(len(text_features)):
        similarities = similarity_matrix[:, i]
        rank = (similarities >= similarities[i]).sum().item()
        t2a_ranks.append(rank)
    
    # Compute recall metrics
    a2t_r1 = sum(1 for rank in a2t_ranks if rank == 1) / len(a2t_ranks)
    a2t_r5 = sum(1 for rank in a2t_ranks if rank <= 5) / len(a2t_ranks)
    a2t_r10 = sum(1 for rank in a2t_ranks if rank <= 10) / len(a2t_ranks)
    
    t2a_r1 = sum(1 for rank in t2a_ranks if rank == 1) / len(t2a_ranks)
    t2a_r5 = sum(1 for rank in t2a_ranks if rank <= 5) / len(t2a_ranks)
    t2a_r10 = sum(1 for rank in t2a_ranks if rank <= 10) / len(t2a_ranks)
    
    return {
        'audio_to_text': {'R@1': a2t_r1, 'R@5': a2t_r5, 'R@10': a2t_r10},
        'text_to_audio': {'R@1': t2a_r1, 'R@5': t2a_r5, 'R@10': t2a_r10}
    }