"""
Dynamic adaptation of special token embeddings

Extracted from: part4/chapter10/embedding_design.tex
Block: 4
Lines: 61
"""

class DynamicEmbeddingAdapter:
    def __init__(self, model, special_token_ids):
        self.model = model
        self.special_token_ids = special_token_ids
        self.usage_statistics = {tid: {'count': 0, 'contexts': []} 
                                 for tid in special_token_ids}
        
    def track_usage(self, input_ids, attention_weights):
        """Track how special tokens are being used."""
        batch_size, seq_len = input_ids.shape
        
        for token_id in self.special_token_ids:
            # Find positions of special token
            positions = (input_ids == token_id).nonzero(as_tuple=True)
            
            if len(positions[0]) > 0:
                for batch_idx, pos_idx in zip(positions[0], positions[1]):
                    self.usage_statistics[token_id]['count'] += 1
                    
                    # Store attention context
                    token_attention = attention_weights[batch_idx, :, pos_idx, :]
                    avg_attention = token_attention.mean(dim=0)  # Average over heads
                    self.usage_statistics[token_id]['contexts'].append(avg_attention)
                    
    def adapt_embeddings(self, adaptation_rate=0.01):
        """Adapt embeddings based on usage patterns."""
        embeddings = self.model.embeddings.word_embeddings
        
        for token_id in self.special_token_ids:
            stats = self.usage_statistics[token_id]
            
            if stats['count'] > 100:  # Sufficient usage for adaptation
                # Analyze attention patterns
                contexts = torch.stack(stats['contexts'][-100:])  # Last 100 uses
                
                # Compute principal components of attention patterns
                U, S, V = torch.svd(contexts.T)
                principal_direction = U[:, 0]  # First principal component
                
                # Get tokens that receive most attention from this special token
                top_attended_positions = principal_direction.topk(10).indices
                top_attended_embeddings = embeddings.weight[top_attended_positions]
                
                # Adapt embedding toward attended context
                context_centroid = top_attended_embeddings.mean(dim=0)
                current_embedding = embeddings.weight[token_id]
                
                # Gradual adaptation
                adapted_embedding = ((1 - adaptation_rate) * current_embedding + 
                                   adaptation_rate * context_centroid)
                
                embeddings.weight.data[token_id] = adapted_embedding
                
                # Reset statistics periodically
                if stats['count'] > 1000:
                    stats['count'] = 0
                    stats['contexts'] = stats['contexts'][-100:]  # Keep recent history
                    
    def reinforcement_adaptation(self, token_id, reward_signal):
        """Adapt embedding based on task performance feedback."""
        embeddings = self.model.embeddings.word_embeddings
        current_embedding = embeddings.weight[token_id]
        
        # Compute update direction based on reward
        if reward_signal > 0:
            # Positive reward: reinforce current direction
            noise = torch.randn_like(current_embedding) * 0.01
            update = current_embedding + noise
        else:
            # Negative reward: explore different direction
            noise = torch.randn_like(current_embedding) * 0.05
            update = current_embedding - reward_signal * noise
            
        # Apply update with learning rate
        learning_rate = 0.001 * abs(reward_signal)
        new_embedding = (1 - learning_rate) * current_embedding + learning_rate * update
        
        embeddings.weight.data[token_id] = new_embedding