"""
Language switching tokens for multi-language code generation

Extracted from: part2/chapter06/code_generation.tex
Block: 1
Lines: 61
"""

class MultiLanguageCodeTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, num_languages=10):
        super().__init__()
        
        # Base transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=12,
                batch_first=True
            ),
            num_layers=12
        )
        
        # Language-specific embeddings
        self.language_embeddings = nn.Embedding(num_languages, embed_dim)
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Language switching tokens
        self.language_switch_tokens = nn.ParameterDict({
            'python': nn.Parameter(torch.randn(1, embed_dim)),
            'javascript': nn.Parameter(torch.randn(1, embed_dim)),
            'java': nn.Parameter(torch.randn(1, embed_dim)),
            'cpp': nn.Parameter(torch.randn(1, embed_dim)),
            'rust': nn.Parameter(torch.randn(1, embed_dim)),
        })
        
        # Language-specific code heads
        self.language_heads = nn.ModuleDict({
            lang: nn.Linear(embed_dim, vocab_size) 
            for lang in self.language_switch_tokens.keys()
        })
        
    def forward(self, input_ids, language_ids):
        # Token embeddings
        token_embeds = self.token_embeddings(input_ids)
        
        # Language embeddings
        lang_embeds = self.language_embeddings(language_ids)
        
        # Combine embeddings
        combined_embeds = token_embeds + lang_embeds
        
        # Add language switch tokens at appropriate positions
        enhanced_embeds = self.add_language_switches(combined_embeds, language_ids)
        
        # Transformer processing
        output = self.transformer(enhanced_embeds)
        
        return output
    
    def add_language_switches(self, embeddings, language_ids):
        """Add language switch tokens at language transition points."""
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Detect language transitions
        transitions = (language_ids[:, 1:] != language_ids[:, :-1])
        
        enhanced_embeddings = []
        for b in range(batch_size):
            sequence = [embeddings[b, 0]]  # Start with first token
            
            for i in range(1, seq_len):
                if transitions[b, i-1]:  # Language transition detected
                    new_lang_id = language_ids[b, i].item()
                    lang_name = self.get_language_name(new_lang_id)
                    
                    if lang_name in self.language_switch_tokens:
                        switch_token = self.language_switch_tokens[lang_name]
                        sequence.append(switch_token.squeeze(0))
                
                sequence.append(embeddings[b, i])
            
            # Pad to original length
            while len(sequence) < seq_len:
                sequence.append(torch.zeros(embed_dim, device=embeddings.device))
            
            enhanced_embeddings.append(torch.stack(sequence[:seq_len]))
        
        return torch.stack(enhanced_embeddings)