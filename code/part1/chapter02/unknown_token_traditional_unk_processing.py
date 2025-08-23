"""
Traditional UNK Processing

Extracted from: part1/chapter02/unknown_token.tex
Block: 1
Lines: 56
"""

class TraditionalTokenizer:
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.unk_token = "[UNK]"
        self.unk_id = 0
        
    def build_vocab(self, texts):
        # Count word frequencies
        word_counts = {}
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and take top K
        sorted_words = sorted(word_counts.items(), 
                            key=lambda x: x[1], reverse=True)
        
        # Build vocabulary
        self.word_to_id[self.unk_token] = self.unk_id
        self.id_to_word[self.unk_id] = self.unk_token
        
        for i, (word, count) in enumerate(sorted_words[:self.vocab_size-1]):
            word_id = i + 1
            self.word_to_id[word] = word_id
            self.id_to_word[word_id] = word
            
    def encode(self, text):
        tokens = []
        for word in text.split():
            if word in self.word_to_id:
                tokens.append(self.word_to_id[word])
            else:
                tokens.append(self.unk_id)  # Map to UNK
        return tokens
    
    def decode(self, token_ids):
        words = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                words.append(self.id_to_word[token_id])
            else:
                words.append(self.unk_token)
        return " ".join(words)

# Example usage
tokenizer = TraditionalTokenizer(vocab_size=1000)

# Build vocabulary from training data
training_texts = [
    "the quick brown fox jumps over the lazy dog",
    "natural language processing is fascinating",
    "transformers revolutionized machine learning"
]
tokenizer.build_vocab(training_texts)

# Handle OOV words
test_text = "the sophisticated algorithm demonstrates remarkable performance"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)

print(f"Original: {test_text}")
print(f"Encoded:  {encoded}")
print(f"Decoded:  {decoded}")
# Output might be: "the [UNK] [UNK] [UNK] [UNK] [UNK]"