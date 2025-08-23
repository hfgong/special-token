"""
Cross-modal alignment training objectives

Extracted from: part2/chapter05/cross_modal_alignment.tex
Block: 2
Lines: 78
"""

class CrossModalAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07, margin=0.2):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def contrastive_alignment_loss(self, alignment_scores, positive_pairs):
        """Contrastive loss for cross-modal alignment."""
        # alignment_scores: [B, num_alignment_tokens, num_pairs]
        # positive_pairs: [B] indices of positive pairs
        
        batch_size = alignment_scores.shape[0]
        num_tokens = alignment_scores.shape[1]
        
        total_loss = 0
        for token_idx in range(num_tokens):
            scores = alignment_scores[:, token_idx, :]  # [B, num_pairs]
            
            # Create labels for positive pairs
            labels = positive_pairs
            
            # Compute contrastive loss
            loss = F.cross_entropy(scores / self.temperature, labels)
            total_loss += loss
        
        return total_loss / num_tokens
    
    def temporal_alignment_loss(self, alignment_tokens, temporal_labels):
        """Encourage temporal consistency in alignments."""
        # alignment_tokens: [B, seq_len, num_alignment_tokens, embed_dim]
        # temporal_labels: [B, seq_len] time stamps
        
        if alignment_tokens.shape[1] < 2:
            return torch.tensor(0.0, device=alignment_tokens.device)
        
        # Compute temporal smoothness
        temporal_diff = alignment_tokens[:, 1:] - alignment_tokens[:, :-1]
        temporal_penalty = temporal_diff.norm(dim=-1).mean()
        
        return temporal_penalty
    
    def semantic_consistency_loss(self, text_alignments, visual_alignments):
        """Encourage semantic consistency between modality alignments."""
        # Cosine similarity between aligned representations
        text_norm = F.normalize(text_alignments, dim=-1)
        visual_norm = F.normalize(visual_alignments, dim=-1)
        
        similarity = (text_norm * visual_norm).sum(dim=-1)
        
        # Encourage high similarity for aligned content
        consistency_loss = 1 - similarity.mean()
        
        return consistency_loss

def train_aligned_multimodal_model(model, dataloader, optimizer, device):
    """Training loop for aligned multimodal model."""
    
    alignment_loss_fn = CrossModalAlignmentLoss()
    model.train()
    
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        text_ids = batch['text_ids'].to(device)
        images = batch['images'].to(device)
        audio_features = batch['audio_features'].to(device)
        labels = batch['labels'].to(device)
        positive_pairs = batch['positive_pairs'].to(device)
        
        # Forward pass
        outputs = model(
            text_ids=text_ids,
            images=images,
            audio_features=audio_features,
            task='classification'
        )
        
        # Main task loss
        main_loss = F.cross_entropy(outputs['output'], labels)
        
        # Alignment losses
        alignment_outputs = outputs['alignment_outputs']
        
        alignment_loss = 0
        for alignment_output in alignment_outputs:
            if alignment_output['alignment_scores'] is not None:
                align_loss = alignment_loss_fn.contrastive_alignment_loss(
                    alignment_output['alignment_scores'],
                    positive_pairs
                )
                alignment_loss += align_loss
        
        # Total loss
        total_batch_loss = main_loss + 0.1 * alignment_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()
        
        total_loss += total_batch_loss.item()
    
    return total_loss / len(dataloader)