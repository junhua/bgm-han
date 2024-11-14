# BGM-HAN

An open-source implementation of an enhanced Hierarchical Attention Network with Byte-pair Encoded, Gated Residual Connetions and Multihead Attention **(BGM-HAN)** for text classification using PyTorch.

Paper: [ArXiv](https://arxiv.org/pdf/2411.08504)

## Features

- Implements the BGM-HAN model architecture.
- Supports both **BERT** tokenizer and **SentencePiece** tokenizer for text tokenization.
- Includes data processing, model training, and evaluation pipelines.
- Comes with a dummy dataset for testing and demonstration purposes.
- Modular code structure for easy customization and extension.

## Installation

Clone the repository and install the requirements:

```bash
git clone https://github.com/junhua/BGMHAN.git
cd BGMHAN
pip install -r requirements.txt
```

Alternatively, install the package using `setup.py`:

```bash
pip install -e .
```

## Usage

### Preparing the Dummy Dataset

A script to generate a dummy dataset with BPE tokenization is provided for testing purposes. 

```bash
python examples/generate_dummy_data.py
```

This will create a `dummy_dataset.csv` and `tokenizer.model` files in the `data/` directory.

### Running the Example Script

To run the example script with the BERT tokenizer:

```bash
python examples/run_bgmhan.py
```

This script will:

- Load the dummy dataset 
- Perform BPE tokenization and embedding (or BERT tokenization and embedding, if chosen).
- Train the BGM-HAN model.
- Evaluate the model and print results.

### SentencePiece Tokenizer

To use the SentencePiece tokenizer, you need a SentencePiece model file (`.model`). For guidance on training a SentencePiece model, refer to the [sentencepiece documentation](https://github.com/google/sentencepiece).


1. Update the `sentencepiece_model_path` in `examples/run_bgmhan.py` to point to your SentencePiece model file:

   ```python
   sentencepiece_model_path = 'data/sentencepiece.model'  # Update with your actual model path
   ```

2. Change the `tokenizer_type` to `'sentencepiece'` 

3. Run the example script:

   ```bash
   python examples/run_bgmhan.py
   ```

## Requirements

- Python 3.6+
- PyTorch
- Transformers
- SentencePiece (if using SentencePiece tokenizer)
- Other dependencies listed in `requirements.txt`

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Ensure all tests pass before using the package in your projects.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss changes.

### How to Contribute

- **Fork** the repository.
- **Create** a new branch for your feature or bug fix.
- **Commit** your changes with clear messages.
- **Submit** a pull request to the `main` branch.

### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines.
- Use docstrings for all public modules, classes, and methods.
- Run linters like `flake8` or formatters like `black` before committing.

### Reporting Issues

- Use the [GitHub issue tracker](https://github.com/yourusername/BGMHAN/issues) to report bugs or request features.
- Provide as much detail as possible, including steps to reproduce the issue.

## Citation

If you use this code or the BGM-HAN model in your research, please cite our paper:

```bibtex
@article{liu2024bgmhan,
  title={Towards Objective and Unbiased Decision Assessments with LLM-Enhanced Hierarchical Attention Networks},
  author={Liu, Junhua and Lim, Kwan Hui and Lee, Roy Ka-Wei},
  journal={arXiv preprint arXiv:2411.08504},
  year={2024}
}
```
