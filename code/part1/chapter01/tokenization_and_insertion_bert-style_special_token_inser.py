"""
BERT-style special token insertion

Extracted from: part1/chapter01/tokenization_and_insertion.tex
Block: 1
Lines: 55
"""

class BERTTokenizer:
    def __init__(self, vocab, special_tokens):
        self.vocab = vocab
        self.cls_token = special_tokens['CLS']
        self.sep_token = special_tokens['SEP']
        self.pad_token = special_tokens['PAD']
        self.unk_token = special_tokens['UNK']
        self.mask_token = special_tokens['MASK']
        
    def encode_single_sequence(self, text, max_length=512):
        """Encode single sequence with BERT special token pattern"""
        # Step 1: Subword tokenization
        tokens = self.subword_tokenize(text)
        
        # Step 2: Truncate if necessary (reserve space for special tokens)
        if len(tokens) > max_length - 2:
            tokens = tokens[:max_length - 2]
        
        # Step 3: Insert special tokens
        sequence = [self.cls_token] + tokens + [self.sep_token]
        
        # Step 4: Pad to max_length if needed
        while len(sequence) < max_length:
            sequence.append(self.pad_token)
            
        return self.convert_tokens_to_ids(sequence)
    
    def encode_pair_sequence(self, text_a, text_b, max_length=512):
        """Encode sentence pair with BERT special token pattern"""
        tokens_a = self.subword_tokenize(text_a)
        tokens_b = self.subword_tokenize(text_b)
        
        # Reserve space for 3 special tokens: [CLS] text_a [SEP] text_b [SEP]
        max_tokens = max_length - 3
        
        # Truncate sequences proportionally
        if len(tokens_a) + len(tokens_b) > max_tokens:
            tokens_a, tokens_b = self.truncate_sequences(
                tokens_a, tokens_b, max_tokens
            )
        
        # Construct sequence with special tokens
        sequence = ([self.cls_token] + tokens_a + [self.sep_token] + 
                   tokens_b + [self.sep_token])
        
        # Create segment IDs (token type embeddings)
        segment_ids = ([0] * (len(tokens_a) + 2) +  # CLS + text_a + SEP
                      [1] * (len(tokens_b) + 1))       # text_b + SEP
        
        # Pad sequences
        while len(sequence) < max_length:
            sequence.append(self.pad_token)
            segment_ids.append(0)
            
        return {
            'input_ids': self.convert_tokens_to_ids(sequence),
            'token_type_ids': segment_ids,
            'attention_mask': [1 if tok != self.pad_token else 0 for tok in sequence]
        }
    
    def truncate_sequences(self, tokens_a, tokens_b, max_length):
        """Proportionally truncate two sequences to fit max_length"""
        while len(tokens_a) + len(tokens_b) > max_length:
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        return tokens_a, tokens_b