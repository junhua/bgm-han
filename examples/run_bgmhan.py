import torch
from bgmhan import BGMHAN, BGMHANDataProcessor, BGMHANPipeline
import json

if __name__ == "__main__":
    # Define the text columns and target column
    text_columns = ['Field1', 'Field2', 'Field3', 'Field4']
    target_column = 'Target'

    # Choose tokenizer type: 'bert' or 'sentencepiece'
    tokenizer_type = 'sentencepiece' 
    sentencepiece_model_path = '../data/tokenizer.model'  # Update with your actual model path
    
    # Or comment the above and change to 'bert'
    # tokenizer_type = 'bert' 

    

    # Choose model name compatible with tokenizer
    if tokenizer_type == 'bert':
        model_name = 'bert-base-uncased'
    elif tokenizer_type == 'sentencepiece':
        model_name = 'xlm-roberta-base'  # Example model that uses SentencePiece
    else:
        raise ValueError("Invalid tokenizer_type. Choose 'bert' or 'sentencepiece'.")

    # Set up data processor
    data_processor = BGMHANDataProcessor(
        file_path='../data/dummy_dataset.csv',
        text_columns=text_columns,
        target_column=target_column,
        emb_file='embeddings.pkl',
        token_max_length=128,
        model_name=model_name,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        tokenizer_type=tokenizer_type,
        sentencepiece_model_path=sentencepiece_model_path
        # sentencepiece_model_path=None
    )

    # Load data and generate embeddings
    data_processor.load_data()
    data_processor.generate_embeddings(force=False)

    # Initialize model
    bgmhan_model = BGMHAN(
        input_dim=768, 
        hidden_dim=1024,
        num_fields=len(text_columns),
        num_heads=8,
        dropout=0.6
    )

    # Initialize and run pipeline
    pipeline = BGMHANPipeline(data_processor, bgmhan_model, lr=2e-5, ep=100)
    results = pipeline.run()

    # Print results
    print("\nTraining Summary:")
    print(f"Best Validation Accuracy: {results['best_val_acc']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test AUC-ROC: {results.get('test_auc', 'N/A'):.4f}")
    print(f"Test Average Precision: {results.get('test_ap', 'N/A'):.4f}")
    print("\nDetailed classification report:")
    print(json.dumps(results['classification_report'], indent=4))
