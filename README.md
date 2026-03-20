# CAFA-5 Protein Function Prediction

Production-ready ML pipeline for the [Kaggle CAFA-5 Protein Function Prediction competition](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction), predicting Gene Ontology (GO) terms from protein language model embeddings (ESM-2, ProtBERT, T5).

## Project Structure

```
CAFA-5-Protein-Function-Prediction-MLOps/
├── configs/
│   └── config.yaml                # All hyperparams, paths, model selection
├── src/
│   ├── __init__.py
│   ├── config.py                  # YAML config loading + dataclass validation
│   ├── preprocess/
│   │   ├── __init__.py
│   │   ├── dataset.py             # ProteinSequenceDataset (PyTorch Dataset)
│   │   └── preprocessing.py       # Build binary label matrix from train_terms.tsv
│   ├── models/
│   │   ├── __init__.py            # Factory function build_model()
│   │   ├── mlp.py                 # MultiLayerPerceptron
│   │   └── cnn1d.py               # CNN1D
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py             # Training loop, validation, checkpointing
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py           # Load model + generate submission
│   └── utils.py                   # Seed setting, logging setup, device selection
├── scripts/
│   ├── train.py                   # CLI: python scripts/train.py --config configs/config.yaml
│   ├── predict.py                 # CLI: python scripts/predict.py --config configs/config.yaml
│   └── preprocess.py              # CLI: generate label matrix from raw data
├── data/                          # .gitignored; user places data here
├── outputs/                       # .gitignored; checkpoints, logs, submissions
├── notebooks/
│   └── CAFA5-EMS2embeds-Pytorch.ipynb   # Archived original notebook
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

## Background

The Gene Ontology (GO) is a concept hierarchy describing biological function of genes and gene products at different levels of abstraction. This project frames GO term prediction as a **multi-label classification** problem: given a protein embedding, predict which of the top-N GO terms apply.

## Setup

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

### 2. Place data

Download data from the [Kaggle competition page](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/data) and embedding datasets : 
- EMS2 : [cafa-5-ems-2-embeddings-numpy](https://www.kaggle.com/datasets/viktorfairuschin/cafa-5-ems-2-embeddings-numpy)
- ProtBERT: [protbert-embeddings-for-cafa5](https://www.kaggle.com/datasets/henriupton/protbert-embeddings-for-cafa5)
- T5Embeds: [t5embeds](https://www.kaggle.com/datasets/kriukov/t5embeds)

Then organize under `data/`:

```
data/
├── cafa-5-protein-function-prediction/
│   └── Train/
│       ├── train_terms.tsv
│       ├── train_sequences.fasta
│       └── ...
├── cafa-5-ems-2-embeddings-numpy/
│   ├── train_embeddings.npy
│   ├── train_ids.npy
│   ├── test_embeddings.npy
│   └── test_ids.npy
└── ...
```

### 3. Configure

Edit `configs/config.yaml` to adjust paths, model type, hyperparameters, and embedding source.

## Usage

### Preprocess labels

```bash
python scripts/preprocess.py --config configs/config.yaml
```

Generates a binary label matrix (`.npy`) under `outputs/`.

### Train

```bash
python scripts/train.py --config configs/config.yaml
```

Trains the model, saves the best checkpoint (by val F1) to `outputs/checkpoints/best_model.pt`, and writes `outputs/training_history.json`.

### Predict

```bash
python scripts/predict.py --config configs/config.yaml [--checkpoint path/to/model.pt]
```

Produces `outputs/submission.tsv` in CAFA-5 format (Id, GO term, Confidence).

## Configuration

All parameters live in `configs/config.yaml`:

```yaml
data:
  data_dir: "data/cafa-5-protein-function-prediction"
  embeddings_dir: "data"
  embeddings_source: "ESM2"        # ESM2 | ProtBERT | T5
  num_labels: 500
  train_val_split: 0.9

model:
  type: "mlp"                      # mlp | cnn1d
  mlp_hidden_dims: [864, 712]
  cnn_out_channels: [3, 8]
  cnn_kernel_size: 3

training:
  epochs: 5
  batch_size: 128
  learning_rate: 0.001
  scheduler_factor: 0.1
  scheduler_patience: 1
  seed: 42

output:
  output_dir: "outputs"
```

## Models

- **MLP** (`mlp`): Configurable hidden-layer sizes, ReLU activations.
- **CNN1D** (`cnn1d`): Two 1-D conv layers with tanh activations, max pooling, and fully-connected output.

Both output raw logits (no final sigmoid) — `BCEWithLogitsLoss` handles the sigmoid internally for numerical stability.

## License

MIT
