# DA6401 Assignment 3 — Transformer for German-English Machine Translation

This repository contains a from-scratch PyTorch implementation of the encoder-decoder Transformer architecture from **Attention Is All You Need** for German-to-English neural machine translation on the **Multi30k** dataset. The implementation includes custom tokenization, vocabulary construction, masking, scaled dot-product attention, multi-head attention, sinusoidal positional encoding, Noam learning-rate scheduling, label smoothing, greedy decoding, BLEU evaluation, checkpointing, and W&B-based experimental analysis.

---

## Project Structure

```text
.
├── dataset.py          # Multi30k loading, spaCy tokenization, vocabulary building, padding
├── model.py            # Transformer architecture: attention, masks, encoder, decoder, inference
├── lr_scheduler.py     # Noam learning-rate scheduler with warmup and inverse-square-root decay
├── train.py            # Training loop, label smoothing, greedy decoding, BLEU, checkpointing
├── wandb_report.py     # W&B experiments and analysis plots for assignment questions
├── requirements.txt    # Python dependencies
├── checkpoint.pth      # Saved model checkpoint generated after training
└── README.md           # Project documentation
```

---

## Main Features

- **Dataset pipeline:** Uses the `bentrevett/multi30k` dataset with German source sentences and English target sentences.
- **Tokenization:** Uses spaCy tokenizers for German and English.
- **Vocabulary:** Builds source and target vocabularies with special tokens: `<unk>`, `<pad>`, `<sos>`, and `<eos>`.
- **Transformer model:** Implements encoder-decoder Transformer components manually without using `torch.nn.MultiheadAttention`.
- **Attention:** Uses scaled dot-product attention:

```text
Attention(Q, K, V) = softmax(QKᵀ / sqrt(d_k))V
```

- **Masks:** Implements source padding masks and target causal masks.
- **Positional encoding:** Uses sinusoidal positional encoding.
- **Optimization:** Uses Adam with the Noam learning-rate scheduler.
- **Regularization:** Uses dropout, gradient clipping, and label smoothing.
- **Decoding:** Uses greedy autoregressive decoding for inference.
- **Evaluation:** Computes corpus-level BLEU score.
- **Logging:** Uses Weights & Biases for train/validation loss, accuracy, BLEU, attention visualizations, gradient norms, and ablation plots.

---

## Installation

Create and activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

Install the required spaCy language models:

```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```



---

## Training

Run the main training script:

```bash
python train.py
```

The default training configuration is defined inside `train.py`:

```python
d_model = 256
N = 3
num_heads = 8
d_ff = 1024
dropout = 0.3
batch_size = 64
num_epochs = 100
warmup_steps = 4000
smoothing = 0.2
```

During training, the script logs training loss, validation loss, and validation BLEU to W&B. The best model is saved as:

```text
checkpoint.pth
```

The checkpoint stores:

- model weights
- optimizer state
- scheduler state
- model configuration
- source vocabulary
- target vocabulary

---

## Inference

After training, use the saved checkpoint for translation. Depending on the final `model.py` version used, either instantiate directly from the checkpoint or reconstruct the model using the saved configuration.


## Evaluation

The training script evaluates translation quality using corpus BLEU. BLEU is computed using greedy decoding over the validation/test data.

```python
bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
```

A higher BLEU score indicates better overlap between generated translations and reference translations.

R

## Implementation Notes

### Noam Scheduler

The Noam scheduler starts with a warmup phase and then decays the learning rate proportional to the inverse square root of the step number:

```text
lr_scale = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

This stabilizes early Transformer training by avoiding large updates when attention weights are still randomly initialized.

### Label Smoothing

Label smoothing replaces hard one-hot targets with softened target distributions. This prevents the model from becoming over-confident, improves calibration, and usually improves generalization even if the training loss/perplexity is slightly higher.

### Greedy Decoding

Inference is performed autoregressively. The model starts with `<sos>`, predicts one token at a time using `argmax`, and stops when `<eos>` is generated or the maximum length is reached.

---

## Requirements

The main dependencies are:

```text
torch
numpy
matplotlib
scikit-learn
wandb
datasets
spacy
tqdm
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## Expected Outputs

After successful training and analysis, the project should produce:

- `checkpoint.pth` — trained Transformer checkpoint
- W&B logs for training and validation curves
- BLEU score on the test set
- attention heatmaps for the final encoder layer
- comparison plots for scheduler, attention scaling, positional encoding, and label smoothing experiments

## Wandb Report Link
https://api.wandb.ai/links/bt25d030-indian-institute-of-technology-madras/tgh92g6q

## GitHub Repo Link
https://github.com/MaheswarRamS/da6401_assignment_3.git