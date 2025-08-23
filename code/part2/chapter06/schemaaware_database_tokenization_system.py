class DatabaseSchemaTokenizer:
    def __init__(self, base_tokenizer):
        self.base_tokenizer = base_tokenizer
        
        # Database structural tokens
        self.schema_tokens = {
            'TABLE_START': '<TABLE_START>',
            'TABLE_END': '<TABLE_END>',
            'COLUMN_DEF': '<COLUMN_DEF>',
            'PRIMARY_KEY': '<PRIMARY_KEY>',
            'FOREIGN_KEY': '<FOREIGN_KEY>',
            'INDEX': '<INDEX>',
            'CONSTRAINT': '<CONSTRAINT>',
            'RELATIONSHIP': '<RELATIONSHIP>',
            'JOIN': '<JOIN>',
            'SCHEMA_BOUNDARY': '<SCHEMA_BOUNDARY>',
        }
        
        # Data type tokens
        self.data_type_tokens = {
            'INT': '<INT_TYPE>',
            'VARCHAR': '<VARCHAR_TYPE>',
            'TEXT': '<TEXT_TYPE>',
            'DATE': '<DATE_TYPE>',
            'TIMESTAMP': '<TIMESTAMP_TYPE>',
            'BOOLEAN': '<BOOLEAN_TYPE>',
            'DECIMAL': '<DECIMAL_TYPE>',
            'JSON': '<JSON_TYPE>',
        }
        
        # Query operation tokens
        self.query_tokens = {
            'SELECT': '<SELECT_OP>',
            'INSERT': '<INSERT_OP>',
            'UPDATE': '<UPDATE_OP>',
            'DELETE': '<DELETE_OP>',
            'WHERE': '<WHERE_CLAUSE>',
            'GROUP_BY': '<GROUP_BY>',
            'ORDER_BY': '<ORDER_BY>',
            'HAVING': '<HAVING>',
            'SUBQUERY': '<SUBQUERY>',
        }
    
    def tokenize_schema(self, schema_definition):
        """Tokenize database schema definition."""
        tokens = []
        tokens.append(self.schema_tokens['SCHEMA_BOUNDARY'])
        
        for table in schema_definition['tables']:
            tokens.append(self.schema_tokens['TABLE_START'])
            tokens.extend(self.base_tokenizer.tokenize(table['name']))
            
            for column in table['columns']:
                tokens.append(self.schema_tokens['COLUMN_DEF'])
                tokens.extend(self.base_tokenizer.tokenize(column['name']))
                
                if column['type'] in self.data_type_tokens:
                    tokens.append(self.data_type_tokens[column['type']])
                
                if column.get('is_primary_key'):
                    tokens.append(self.schema_tokens['PRIMARY_KEY'])
                
                if column.get('foreign_key'):
                    tokens.append(self.schema_tokens['FOREIGN_KEY'])
                    tokens.extend(self.base_tokenizer.tokenize(
                        column['foreign_key']['table']
                    ))
            
            tokens.append(self.schema_tokens['TABLE_END'])
        
        tokens.append(self.schema_tokens['SCHEMA_BOUNDARY'])
        return tokens
    
    def tokenize_query(self, sql_query):
        """Tokenize SQL query with structure awareness."""
        tokens = []
        query_upper = sql_query.upper()
        
        # Parse query structure
        if 'SELECT' in query_upper:
            tokens.append(self.query_tokens['SELECT'])
        if 'FROM' in query_upper:
            tokens.append(self.schema_tokens['TABLE_START'])
        if 'WHERE' in query_upper:
            tokens.append(self.query_tokens['WHERE'])
        if 'JOIN' in query_upper:
            tokens.append(self.schema_tokens['JOIN'])
        
        # Add actual query tokens
        query_tokens = self.base_tokenizer.tokenize(sql_query)
        tokens.extend(query_tokens)
        
        return tokens

class StructuredDataTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=768):
        super().__init__()
        
        self.schema_tokenizer = DatabaseSchemaTokenizer(None)
        
        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.schema_type_embeddings = nn.Embedding(15, embed_dim)  # Schema element types
        self.relationship_embeddings = nn.Embedding(10, embed_dim)  # Relationship types
        
        # Schema structure encoder
        self.schema_encoder = SchemaStructureEncoder(embed_dim)
        
        # Relationship-aware transformer
        self.transformer = RelationshipAwareTransformer(embed_dim, num_layers=12)
        
        # Output heads
        self.query_head = nn.Linear(embed_dim, vocab_size)
        self.schema_head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, input_ids, schema_structure=None, relationships=None):
        # Token embeddings
        token_embeds = self.token_embeddings(input_ids)
        
        # Add schema structure embeddings
        if schema_structure is not None:
            schema_embeds = self.schema_encoder(schema_structure)
            token_embeds = token_embeds + schema_embeds
        
        # Add relationship embeddings
        if relationships is not None:
            rel_embeds = self.relationship_embeddings(relationships)
            token_embeds = token_embeds + rel_embeds
        
        # Process through relationship-aware transformer
        output = self.transformer(token_embeds, schema_structure)
        
        return output

class SchemaStructureEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        # Table and column embeddings
        self.table_embeddings = nn.Embedding(100, embed_dim)  # Table types
        self.column_embeddings = nn.Embedding(200, embed_dim)  # Column types
        
        # Constraint embeddings
        self.constraint_embeddings = nn.Embedding(20, embed_dim)
        
        # Hierarchical position embeddings
        self.hierarchy_embeddings = nn.Embedding(5, embed_dim)  # Schema levels
        
    def forward(self, schema_structure):
        """Encode database schema structure."""
        if schema_structure is None:
            return None
        
        # Encode table information
        table_embeds = self.table_embeddings(schema_structure['table_ids'])
        
        # Encode column information
        if 'column_ids' in schema_structure:
            column_embeds = self.column_embeddings(schema_structure['column_ids'])
            table_embeds = table_embeds + column_embeds
        
        # Encode constraints
        if 'constraint_ids' in schema_structure:
            constraint_embeds = self.constraint_embeddings(
                schema_structure['constraint_ids']
            )
            table_embeds = table_embeds + constraint_embeds
        
        return table_embeds

class RelationshipAwareTransformer(nn.Module):
    def __init__(self, embed_dim, num_layers=12):
        super().__init__()
        
        self.layers = nn.ModuleList([
            RelationshipAwareLayer(embed_dim) for _ in range(num_layers)
        ])
        
    def forward(self, embeddings, schema_structure=None):
        x = embeddings
        
        for layer in self.layers:
            x = layer(x, schema_structure)
        
        return x

class RelationshipAwareLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        # Standard attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads=12, batch_first=True
        )
        
        # Schema relationship attention
        self.schema_attention = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )
        
        # Feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, schema_structure=None):
        # Self attention
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Schema relationship attention
        if schema_structure is not None:
            schema_mask = self.create_schema_attention_mask(
                schema_structure, x.size(1)
            )
            schema_output, _ = self.schema_attention(x, x, x, attn_mask=schema_mask)
            x = self.norm2(x + schema_output)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        
        return x
    
    def create_schema_attention_mask(self, schema_structure, seq_len):
        """Create attention mask for schema relationships."""
        mask = torch.zeros(seq_len, seq_len)
        
        # Allow attention between related schema elements
        if 'relationships' in schema_structure:
            for source, target in schema_structure['relationships']:
                mask[source, target] = 1
                mask[target, source] = 1  # Bidirectional relationship
        
        return mask
