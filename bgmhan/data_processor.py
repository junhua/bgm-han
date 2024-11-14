import os
import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

class BGMHANDataProcessor:
    def __init__(self, file_path, text_columns, target_column, token_max_length=512, 
                 model_name="bert-base-uncased", emb_path="../data/embeddings", 
                 emb_file="embeddings.pkl", batch_size=32, device="cuda",
                 tokenizer_type="bert", sentencepiece_model_path=None):
        """
        Initialize the data processor.

        Args:
            file_path (str): Path to the CSV file containing data.
            text_columns (list of str): List of text column names to process.
            target_column (str): Name of the target column.
            token_max_length (int): Maximum length of tokens.
            model_name (str): Name of the pre-trained model to use.
            emb_path (str): Path to save embeddings.
            emb_file (str): Name of the embeddings file.
            batch_size (int): Batch size for processing.
            device (str): Device to use ("cuda" or "cpu").
            tokenizer_type (str): Type of tokenizer to use ("bert" or "sentencepiece").
            sentencepiece_model_path (str): Path to the SentencePiece model file (.model).
        """
        self.file_path = file_path
        self.text_columns = list(text_columns) if isinstance(text_columns, (list, tuple)) else [text_columns]
        self.target_column = target_column
        self.model_name = model_name
        self.token_max_length = token_max_length
        self.batch_size = batch_size
        self.tokenizer_type = tokenizer_type
        self.sentencepiece_model_path = sentencepiece_model_path
        print("sentencepiece_model_path", sentencepiece_model_path)
        # Initialize components
        self.device = device
        print(f"Using device: {self.device}")

        # Initialize tokenizer and model
        if self.tokenizer_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name).to(self.device)
        elif self.tokenizer_type == "sentencepiece":
            if not self.sentencepiece_model_path or not os.path.exists(self.sentencepiece_model_path):
                raise ValueError("Valid 'sentencepiece_model_path' must be provided for SentencePiece tokenizer.")
            import sentencepiece as spm
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(self.sentencepiece_model_path)
            # Use an appropriate model that supports SentencePiece tokenization
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
        else:
            raise ValueError("Invalid tokenizer_type. Choose 'bert' or 'sentencepiece'.")

        self.model.eval()  # Set to evaluation mode

        # Initialize other components
        self.label_encoder = LabelEncoder()
        self.emb_path = emb_path
        self.emb_file = emb_file
        os.makedirs(emb_path, exist_ok=True)

        # Initialize dataframes
        self.df = None
        self.original_df = None

    def load_data(self):
        """Load and validate the input data."""
        print("Loading data...")
        self.original_df = pd.read_csv(self.file_path)
        self.df = self.original_df.copy()

        print(f"Loaded data with {len(self.df)} rows")
        print(f"Available columns: {self.df.columns.tolist()}")
        print(f"Target column: {self.target_column}")
        print(f"Text columns: {self.text_columns}")

        # Validate columns
        missing_cols = []
        if self.target_column not in self.df.columns:
            missing_cols.append(self.target_column)

        for col in self.text_columns:
            if col not in self.df.columns:
                missing_cols.append(col)

        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")

        # Fill NaN values in text columns
        for col in self.text_columns:
            self.df[col] = self.df[col].fillna("None")

        print("Successfully loaded and validated all columns")
        return True

    def generate_batch_embeddings(self, texts):
        """Generate embeddings for a batch of texts using the selected tokenizer."""
        if self.tokenizer_type == "bert":
            # Tokenize with BertTokenizer
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.token_max_length,
                return_tensors='pt'
            )

            # Move to device
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

        elif self.tokenizer_type == "sentencepiece":
            # Tokenize with SentencePiece
            input_ids = []
            attention_mask = []
            for text in texts:
                tokens = self.tokenizer.encode(text)
                tokens = tokens[:self.token_max_length]  # Truncate if necessary
                padding_length = self.token_max_length - len(tokens)
                input_ids.append(tokens + [0]*padding_length)
                attention_mask.append([1]*len(tokens) + [0]*padding_length)

            # Convert to tensors
            input_ids = torch.tensor(input_ids).to(self.device)
            attention_mask = torch.tensor(attention_mask).to(self.device)

        else:
            raise ValueError("Invalid tokenizer_type. Choose 'bert' or 'sentencepiece'.")

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            # Get the [CLS] token embeddings
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings

    def generate_embeddings(self, force=False):
        """Generate embeddings for all text columns."""
        if not force and self.load_embeddings():
            print("Using existing embeddings")
            return

        print("Generating new embeddings...")

        # Process each text column
        for idx, col in enumerate(self.text_columns, 1):
            print(f"\nProcessing column {idx}/{len(self.text_columns)}: {col}")
            embedding_col = f"{col}_embedding"

            # Process in batches
            all_embeddings = []
            for i in tqdm(range(0, len(self.df), self.batch_size), 
                          desc=f"Generating embeddings for {col}"):
                batch_texts = self.df[col].iloc[i:i + self.batch_size].tolist()
                batch_embeddings = self.generate_batch_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)

            # Store embeddings in dataframe
            self.df[embedding_col] = all_embeddings

        # Save embeddings
        self.save_embeddings()
        print("\nFinished generating all embeddings")

    def save_embeddings(self):
        """Save embeddings and data to file."""
        print("Saving embeddings and data...")
        save_path = os.path.join(self.emb_path, self.emb_file)

        # Create a DataFrame with embeddings and original data
        save_df = pd.DataFrame()

        # Add embeddings
        for col in self.text_columns:
            save_df[f"{col}_embedding"] = self.df[f"{col}_embedding"]

        # Add original columns
        save_df[self.target_column] = self.original_df[self.target_column]
        save_df[self.text_columns] = self.original_df[self.text_columns]

        # Save to pickle file
        save_df.to_pickle(save_path)
        print(f"Saved data to {save_path}")
        print(f"Saved columns: {save_df.columns.tolist()}")

    def load_embeddings(self):
        """Load embeddings from file if they exist."""
        embeddings_path = os.path.join(self.emb_path, self.emb_file)

        if os.path.exists(embeddings_path):
            print("Loading existing embeddings...")
            loaded_df = pd.read_pickle(embeddings_path)

            # Verify all required columns exist
            required_cols = ([f"{col}_embedding" for col in self.text_columns] + 
                             [self.target_column] + self.text_columns)

            missing_cols = [col for col in required_cols if col not in loaded_df.columns]

            if not missing_cols:
                self.df = loaded_df
                self.original_df = loaded_df
                print("Successfully loaded existing embeddings")
                print(f"Loaded columns: {self.df.columns.tolist()}")
                return True
            else:
                print(f"Missing columns in saved data: {missing_cols}")
                return False

        print("No existing embeddings file found")
        return False

    def prepare_data(self):
        """Prepare data for model training."""
        print("Preparing data for model...")
        print(f"Available columns: {self.df.columns.tolist()}")

        # Verify all required columns exist
        required_cols = [f"{col}_embedding" for col in self.text_columns] + [self.target_column]
        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            raise KeyError(f"Missing columns: {missing_cols}. Available columns: {self.df.columns.tolist()}")

        # Stack embeddings and prepare target
        X_embeddings = {col: np.stack(self.df[f"{col}_embedding"].values) 
                        for col in self.text_columns}
        y = self.label_encoder.fit_transform(self.df[self.target_column])

        print(f"Prepared {len(self.text_columns)} feature columns and target column '{self.target_column}'")
        print(f"Target classes: {self.label_encoder.classes_}")

        return X_embeddings, y, self.label_encoder.classes_
