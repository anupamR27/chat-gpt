# Mini GPT – Transformer Language Model

A character-level GPT-style language model built from scratch using PyTorch and trained on the Tiny Shakespeare dataset.

This project follows Andrej Karpathy's "Let's build GPT: from scratch" approach, with improvements including checkpointing, interactive text generation, and PyTorch's Scaled Dot-Product Attention.

## Features

- Character-level tokenizer
- Token embeddings
- Positional embeddings
- Causal self-attention
- Multi-head self-attention
- Scaled Dot-Product Attention
- Transformer blocks
- Residual connections
- Layer normalization
- Feed-forward neural networks
- Dropout regularization
- Temperature-based text generation
- Training and validation loss evaluation
- Model checkpointing
- Interactive prompt-based generation
- Apple Silicon MPS acceleration

## Model Architecture

| Component | Value |
|---|---:|
| Parameters | ~0.825M |
| Embedding Dimension | 128 |
| Attention Heads | 4 |
| Transformer Layers | 4 |
| Context Length | 128 characters |
| Batch Size | 64 |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Training Iterations | 5000 |

## Architecture

```text
Input Characters
       ↓
Character Embeddings + Positional Embeddings
       ↓
4 × Transformer Blocks
       ↓
Layer Normalization
       ↓
Linear Language Model Head
       ↓
Next Character Prediction
````

Each Transformer block contains:

```text
Input
  ↓
LayerNorm
  ↓
Multi-Head Causal Self-Attention
  ↓
Residual Connection
  ↓
LayerNorm
  ↓
Feed-Forward Network
  ↓
Residual Connection
```

## Training Results

The model was trained on the Tiny Shakespeare dataset for 5000 iterations.

Final results:

```text
Train Loss: 1.5097
Validation Loss: 1.6936
```

Model size:

```text
0.824897 M parameters
```

## Running the Project

Install PyTorch:

```bash
pip install torch
```

Run:

```bash
python bigram.py
```

If `checkpoint.pt` does not exist, the model trains automatically and saves the trained weights.

On future runs, the model loads the checkpoint without retraining.

## Interactive Generation

After loading the trained model:

```text
Enter a prompt (or type 'quit'):
```

Example:

```text
Enter a prompt (or type 'quit'): ROMEO:
```

The model then generates text conditioned on your prompt.

Type:

```text
quit
```

to exit.

## Temperature

Generation randomness can be controlled using:

```python
temperature=0.5
```

* Lower temperature → more predictable output
* Higher temperature → more random output

## Technologies Used

* Python
* PyTorch
* Transformer Architecture
* Self-Attention
* Apple Silicon MPS

## Dataset

Tiny Shakespeare dataset:

[https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

## Inspiration

Based on Andrej Karpathy's:

"Let's build GPT: from scratch, in code, spelled out."

[https://www.youtube.com/watch?v=kCc8FmEb1Y](https://www.youtube.com/watch?v=kCc8FmEb1Y)

## Future Improvements

* Subword tokenization
* Top-k and top-p sampling
* Key-value caching
* Faster inference
* Web interface
* Larger model architecture
* Fine-tuning on custom datasets

