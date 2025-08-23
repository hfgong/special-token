class MathematicalTokenizer:
    def __init__(self, base_tokenizer):
        self.base_tokenizer = base_tokenizer
        
        # Mathematical special tokens
        self.math_tokens = {
            'FORMULA_START': '<FORMULA_START>',
            'FORMULA_END': '<FORMULA_END>',
            'EQUATION_START': '<EQ_START>',
            'EQUATION_END': '<EQ_END>',
            'MATRIX_START': '<MATRIX_START>',
            'MATRIX_END': '<MATRIX_END>',
            'INTEGRAL': '<INTEGRAL>',
            'SUMMATION': '<SUM>',
            'DERIVATIVE': '<DERIVATIVE>',
            'FRACTION': '<FRACTION>',
            'SUBSCRIPT': '<SUB>',
            'SUPERSCRIPT': '<SUP>',
            'SQRT': '<SQRT>',
            'UNITS': '<UNITS>',
        }
        
        # Mathematical operators and symbols
        self.math_operators = {
            '+': '<PLUS>', '-': '<MINUS>', '*': '<MULT>', '/': '<DIV>',
            '=': '<EQUALS>', '<': '<LESS>', '>': '<GREATER>',
            '$\\leq$': '<LEQ>', '$\\geq$': '<GEQ>', '$\\neq$': '<NEQ>',
            '$\\partial$': '<PARTIAL>', '$\\nabla$': '<GRADIENT>', '$\\int$': '<INTEGRAL_SYM>',
            '$\\sum$': '<SUM_SYM>', '$\\prod$': '<PROD>', '$\\sqrt{}$': '<SQRT_SYM>',
            '$\\alpha$': '<ALPHA>', '$\\beta$': '<BETA>', '$\\gamma$': '<GAMMA>',
            '$\\delta$': '<DELTA>', '$\\epsilon$': '<EPSILON>', '$\\theta$': '<THETA>',
            '$\\lambda$': '<LAMBDA>', '$\\mu$': '<MU>', '$\\pi$': '<PI>', '$\\sigma$': '<SIGMA>',
        }
    
    def tokenize_scientific_text(self, text):
        """Tokenize text containing mathematical expressions."""
        tokens = []
        i = 0
        
        while i < len(text):
            # Detect LaTeX math expressions
            if text[i:i+2] == '$$':
                tokens.append(self.math_tokens['EQUATION_START'])
                i += 2
                start = i
                
                # Find end of equation
                while i < len(text) - 1 and text[i:i+2] != '$$':
                    i += 1
                
                # Tokenize math content
                math_content = text[start:i]
                math_tokens = self.tokenize_math_expression(math_content)
                tokens.extend(math_tokens)
                
                tokens.append(self.math_tokens['EQUATION_END'])
                i += 2
                
            elif text[i] == '$':
                tokens.append(self.math_tokens['FORMULA_START'])
                i += 1
                start = i
                
                # Find end of inline formula
                while i < len(text) and text[i] != '$':
                    i += 1
                
                # Tokenize math content
                math_content = text[start:i]
                math_tokens = self.tokenize_math_expression(math_content)
                tokens.extend(math_tokens)
                
                tokens.append(self.math_tokens['FORMULA_END'])
                i += 1
                
            else:
                # Regular text
                char = text[i]
                if char in self.math_operators:
                    tokens.append(self.math_operators[char])
                else:
                    tokens.append(char)
                i += 1
        
        return tokens
    
    def tokenize_math_expression(self, math_expr):
        """Tokenize a mathematical expression."""
        tokens = []
        i = 0
        
        while i < len(math_expr):
            # Handle fractions
            if math_expr[i:i+5] == '\\frac':
                tokens.append(self.math_tokens['FRACTION'])
                i += 5
                continue
            
            # Handle integrals
            if math_expr[i:i+4] == '\\int':
                tokens.append(self.math_tokens['INTEGRAL'])
                i += 4
                continue
            
            # Handle summations
            if math_expr[i:i+4] == '\\sum':
                tokens.append(self.math_tokens['SUMMATION'])
                i += 4
                continue
            
            # Handle square roots
            if math_expr[i:i+5] == '\\sqrt':
                tokens.append(self.math_tokens['SQRT'])
                i += 5
                continue
            
            # Handle subscripts
            if math_expr[i] == '_':
                tokens.append(self.math_tokens['SUBSCRIPT'])
                i += 1
                continue
            
            # Handle superscripts
            if math_expr[i] == '^':
                tokens.append(self.math_tokens['SUPERSCRIPT'])
                i += 1
                continue
            
            # Handle matrices
            if math_expr[i:i+7] == '\\matrix':
                tokens.append(self.math_tokens['MATRIX_START'])
                i += 7
                continue
            
            # Regular character or operator
            char = math_expr[i]
            if char in self.math_operators:
                tokens.append(self.math_operators[char])
            else:
                tokens.append(char)
            i += 1
        
        return tokens

class ScientificTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=768):
        super().__init__()
        
        self.tokenizer = MathematicalTokenizer(None)
        
        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.math_type_embeddings = nn.Embedding(20, embed_dim)  # Different math contexts
        
        # Mathematical structure encoder
        self.math_structure_encoder = MathStructureEncoder(embed_dim)
        
        # Transformer with math-aware attention
        self.transformer = MathAwareTransformer(embed_dim, num_layers=12)
        
        # Output heads
        self.text_head = nn.Linear(embed_dim, vocab_size)
        self.math_head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, input_ids, math_structure=None, math_context=None):
        # Token embeddings
        token_embeds = self.token_embeddings(input_ids)
        
        # Add mathematical context embeddings
        if math_context is not None:
            math_embeds = self.math_type_embeddings(math_context)
            token_embeds = token_embeds + math_embeds
        
        # Encode mathematical structure
        if math_structure is not None:
            structure_embeds = self.math_structure_encoder(math_structure)
            token_embeds = token_embeds + structure_embeds
        
        # Process through math-aware transformer
        output = self.transformer(token_embeds, math_structure)
        
        return output

class MathStructureEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        # Structure type embeddings
        self.structure_types = nn.Embedding(10, embed_dim)  # fraction, integral, etc.
        
        # Hierarchical position embeddings
        self.hierarchy_embeddings = nn.Embedding(8, embed_dim)  # nested levels
        
    def forward(self, math_structure):
        """Encode mathematical structure information."""
        if math_structure is None:
            return None
        
        structure_embeds = self.structure_types(math_structure['types'])
        
        if 'hierarchy' in math_structure:
            hierarchy_embeds = self.hierarchy_embeddings(math_structure['hierarchy'])
            structure_embeds = structure_embeds + hierarchy_embeds
        
        return structure_embeds

class MathAwareTransformer(nn.Module):
    def __init__(self, embed_dim, num_layers=12):
        super().__init__()
        
        self.layers = nn.ModuleList([
            MathAwareLayer(embed_dim) for _ in range(num_layers)
        ])
        
    def forward(self, embeddings, math_structure=None):
        x = embeddings
        
        for layer in self.layers:
            x = layer(x, math_structure)
        
        return x

class MathAwareLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        # Standard attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads=12, batch_first=True
        )
        
        # Mathematical structure attention
        self.math_attention = nn.MultiheadAttention(
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
        
    def forward(self, x, math_structure=None):
        # Self attention
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Mathematical structure attention
        if math_structure is not None:
            math_mask = self.create_math_attention_mask(math_structure, x.size(1))
            math_output, _ = self.math_attention(x, x, x, attn_mask=math_mask)
            x = self.norm2(x + math_output)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        
        return x
    
    def create_math_attention_mask(self, math_structure, seq_len):
        """Create attention mask for mathematical expressions."""
        mask = torch.zeros(seq_len, seq_len)
        
        # Allow attention within same mathematical expression
        if 'boundaries' in math_structure:
            for start, end in math_structure['boundaries']:
                mask[start:end, start:end] = 1
        
        return mask
