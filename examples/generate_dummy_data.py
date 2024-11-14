import pandas as pd
import numpy as np
import sentencepiece as spm
import os

# Number of samples to generate
num_samples = 100  # Adjust as needed
classes = ['Class1', 'Class2']

# Generate sample data
data = {
    'Field1': ['Sample text for field1 sample {}'.format(i) for i in range(num_samples)],
    'Field2': ['Sample text for field2 sample {}'.format(i) for i in range(num_samples)],
    'Field3': ['Sample text for field3 sample {}'.format(i) for i in range(num_samples)],
    'Field4': ['Sample text for field4 sample {}'.format(i) for i in range(num_samples)],
    'Target': np.random.choice(classes, num_samples)
}

# Create a DataFrame and save as CSV
df = pd.DataFrame(data)
os.makedirs('data', exist_ok=True)
df.to_csv('data/dummy_dataset.csv', index=False)
print("Dummy dataset saved to 'data/dummy_dataset.csv'.")

# Combine all text fields into a single text corpus for tokenizer training
text_corpus_path = 'data/text_corpus.txt'
with open(text_corpus_path, 'w') as f:
    for field in ['Field1', 'Field2', 'Field3', 'Field4']:
        f.write("\n".join(df[field].values) + "\n")

# Train a SentencePiece model
spm.SentencePieceTrainer.train(
    input=text_corpus_path, 
    model_prefix='data/tokenizer', 
    vocab_size = 28,  # Adjust vocabulary size as needed (i.e. 5000)
    model_type='bpe'  # Use BPE (Byte-Pair Encoding)
)

print("SentencePiece model generated as 'data/tokenizer.model'.")
