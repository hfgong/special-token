"""
Natural language to SQL generation system

Extracted from: part2/chapter06/structured_data.tex
Block: 3
Lines: 57
"""

class NL2SQLTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=768):
        super().__init__()
        
        self.data_transformer = DataPipelineTransformer(vocab_size, embed_dim)
        
        # Natural language encoder
        self.nl_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead=12, batch_first=True),
            num_layers=6
        )
        
        # SQL decoder
        self.sql_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_dim, nhead=12, batch_first=True),
            num_layers=6
        )
        
        # Schema-aware attention
        self.schema_attention = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )
        
        # Query optimization head
        self.query_optimizer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, vocab_size)
        )
        
    def forward(self, nl_query, schema_context, target_sql=None):
        # Encode natural language query
        nl_encoded = self.nl_encoder(nl_query)
        
        # Encode schema context
        schema_encoded = self.data_transformer(schema_context)
        
        # Schema-aware attention
        query_context, _ = self.schema_attention(
            nl_encoded, schema_encoded, schema_encoded
        )
        
        if target_sql is not None:
            # Training mode: generate SQL with teacher forcing
            sql_output = self.sql_decoder(target_sql, query_context)
        else:
            # Inference mode: generate SQL autoregressively
            sql_output = self.generate_sql(query_context)
        
        # Optimize generated query
        optimized_sql = self.query_optimizer(sql_output)
        
        return optimized_sql
    
    def generate_sql(self, query_context, max_length=200):
        """Generate SQL query autoregressively."""
        batch_size = query_context.size(0)
        device = query_context.device
        
        # Start with special token
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        for i in range(max_length):
            # Decode next token
            output = self.sql_decoder(generated, query_context)
            next_token = torch.argmax(output[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for end token
            if torch.all(next_token == 2):  # Assuming 2 is end token
                break
        
        return generated