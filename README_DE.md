# CAFA-5 Proteinfunktions-Vorhersage — MLOps Pipeline

Eine produktionsreife MLOps-Pipeline für den [Kaggle CAFA-5 Wettbewerb](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction). Sie sagt Gene Ontology (GO) Terme aus Protein-Sprachmodell-Embeddings (ESM-2, ProtBERT, T5) vorher.

## Projektziele

| Ziel | Metrik | Zielwert |
|---|---|---|
| GO-Terme pro Protein vorhersagen | F1-Score (micro) | > 0.10 Baseline |
| Multi-Label-Klassifikation | Val F1 | Bestes Checkpoint gespeichert |
| Reproduzierbare Pipeline | Geseedete Splits | seed=42 |

## Architektur

```
data/embeddings/          data/cafa-5-.../Train/
       │                          │
       ▼                          ▼
  preprocess ──────────► label_matrix_top500/
       │
       ▼
    train ──────────────► outputs/checkpoints/best_model.pt
       │
       ▼
   predict ─────────────► outputs/submission.tsv
       │
       ▼
  Inference API ──────────► http://localhost:8000
```

## Projektstruktur

```
CAFA-5-MLOps/
├── configs/
│   └── config.yaml              # Alle Hyperparameter und Pfade
├── docker/
│   ├── Dockerfile.amd           # AMD ROCm (RX 9060 XT, gfx1200)
│   ├── Dockerfile.nvidia        # NVIDIA CUDA
│   ├── Dockerfile.cpu           # CPU-only Fallback
│   └── Dockerfile.api           # Leichtgewichtige Inference API
├── scripts/
│   ├── preprocess.py            # Binäre Label-Matrix erstellen
│   ├── train.py                 # Modell trainieren
│   └── predict.py               # submission.tsv generieren
├── src/
│   ├── api/
│   │   └── app.py               # FastAPI Inference API
│   ├── data/
│   │   ├── dataset.py           # PyTorch Dataset
│   │   └── preprocessing.py    # Label-Matrix-Builder
│   ├── inference/
│   │   └── predictor.py        # Checkpoint laden + Vorhersagen
│   ├── models/
│   │   ├── mlp.py               # MultiLayerPerceptron
│   │   └── cnn1d.py             # 1D Convolutional Network
│   ├── training/
│   │   └── trainer.py           # Trainingsschleife + Checkpointing
│   ├── config.py                # YAML-Config-Loader
│   └── utils.py                 # Seed, Device, Logger
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

## Schnellstart

### Voraussetzungen
- Docker + Docker Compose
- AMD GPU (ROCm) / NVIDIA GPU (CUDA) / CPU

### 1. Repo klonen und Daten vorbereiten

```bash
git clone https://github.com/SimonCardia/CAFA-5-MLOps.git
cd CAFA-5-MLOps
mkdir -p data/embeddings/esm2 data/cafa-5-protein-function-prediction outputs

# Von Kaggle herunterladen
kaggle datasets download viktorfairuschin/cafa-5-ems-2-embeddings-numpy \
  --unzip -p data/embeddings/esm2/
kaggle datasets download siddhvr/train-targets-top500 \
  --unzip -p outputs/label_matrix_top500/
# train_terms.tsv in data/cafa-5-protein-function-prediction/Train/ ablegen
```

### 2. Bauen und starten

```bash
# AMD GPU
docker compose --profile amd build
docker compose --profile amd up

# NVIDIA GPU
docker compose --profile nvidia build
docker compose --profile nvidia up

# Nur CPU
docker compose --profile cpu build
docker compose --profile cpu up
```

### 3. Inference API starten

```bash
docker compose --profile amd build api
docker compose --profile amd up api
# API erreichbar unter: http://localhost:8000
# Swagger-Docs unter:   http://localhost:8000/docs
```

## Modelle

| Modell | Parameter | Beschreibung |
|---|---|---|
| `cnn1d` | 2,6M | Zwei 1D-Faltungsschichten + Max-Pooling + FC |
| `mlp` | ~1,2M | Drei vollverbundene Schichten mit ReLU |

Beide geben rohe Logits aus — `BCEWithLogitsLoss` wendet Sigmoid intern an.

## Ergebnisse (Baseline)

| Epoche | Train F1 | Val F1 |
|---|---|---|
| 1 | 0,019 | 0,047 |
| 2 | 0,069 | 0,079 |
| 3 | 0,097 | 0,108 |
| 4 | 0,117 | 0,117 |
| 5 | 0,137 | **0,129** |

Modell: CNN1D, Embeddings: ESM2, 5 Epochen, AMD RX 9060 XT

## Konfiguration

Alle Parameter in `configs/config.yaml`:

```yaml
data:
  embeddings_source: "ESM2"   # ESM2 | ProtBERT | T5
  num_labels: 500
  train_val_split: 0.9
model:
  type: "cnn1d"               # mlp | cnn1d
training:
  epochs: 5
  batch_size: 128
  learning_rate: 0.001
  seed: 42
```

## Lizenz

MIT
