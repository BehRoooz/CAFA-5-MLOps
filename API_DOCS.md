# API Documentation / API-Dokumentation

## English

### Base URL
```
http://localhost:8000
```

### Endpoints

---

#### GET /health

Returns the current status of the API and loaded model.

**Response:**
```json
{
  "status": "ok",
  "model": "cnn1d",
  "embedding_dim": 1280,
  "num_labels": 500,
  "device": "cpu"
}
```

---

#### POST /predict

Predicts GO terms for a single protein embedding.

**Request body:**
```json
{
  "embedding": [0.12, -0.87, 0.45, ...],
  "threshold": 0.5
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `embedding` | float[] | Yes | Protein embedding vector (1280-dim for ESM2) |
| `threshold` | float | No (default: 0.5) | Minimum confidence score (0.0–1.0) |

**Response:**
```json
{
  "num_predictions": 3,
  "predictions": [
    {"go_term": "GO:0008150", "confidence": 0.953},
    {"go_term": "GO:0003674", "confidence": 0.797},
    {"go_term": "GO:0005575", "confidence": 0.712}
  ]
}
```

**Error responses:**

| Code | Description |
|---|---|
| 422 | Embedding dimension mismatch |
| 500 | Model not loaded |

---

#### GET /terms

Returns all GO terms the model was trained on.

**Response:**
```json
{
  "num_terms": 500,
  "terms": ["GO:0005575", "GO:0008150", ...]
}
```

---

### Example — Python

```python
import json
import urllib.request
import numpy as np

# Load a real embedding
embedding = np.load("data/embeddings/esm2/train_embeddings.npy")[0].tolist()

data = json.dumps({
    "embedding": embedding,
    "threshold": 0.5
}).encode()

req = urllib.request.Request(
    "http://localhost:8000/predict",
    data=data,
    headers={"Content-Type": "application/json"}
)
response = urllib.request.urlopen(req)
result = json.loads(response.read())
print(f"Predicted {result['num_predictions']} GO terms")
for pred in result["predictions"][:5]:
    print(f"  {pred['go_term']}: {pred['confidence']:.3f}")
```

### Example — curl

```bash
# Health check
curl http://localhost:8000/health

# Predict with random embedding
python3 -c "
import json, random, urllib.request
emb = [random.uniform(-1,1) for _ in range(1280)]
data = json.dumps({'embedding': emb, 'threshold': 0.7}).encode()
req = urllib.request.Request('http://localhost:8000/predict', data=data,
      headers={'Content-Type': 'application/json'})
print(urllib.request.urlopen(req).read().decode())
"
```

### Interactive Docs

Open in browser: `http://localhost:8000/docs`

---

---

## Deutsch

### Basis-URL
```
http://localhost:8000
```

### Endpunkte

---

#### GET /health

Gibt den aktuellen Status der API und des geladenen Modells zurück.

**Antwort:**
```json
{
  "status": "ok",
  "model": "cnn1d",
  "embedding_dim": 1280,
  "num_labels": 500,
  "device": "cpu"
}
```

---

#### POST /predict

Sagt GO-Terme für ein einzelnes Protein-Embedding vorher.

**Anfrage-Body:**
```json
{
  "embedding": [0.12, -0.87, 0.45, ...],
  "threshold": 0.5
}
```

| Feld | Typ | Pflicht | Beschreibung |
|---|---|---|---|
| `embedding` | float[] | Ja | Protein-Embedding-Vektor (1280-dim für ESM2) |
| `threshold` | float | Nein (Standard: 0.5) | Minimale Konfidenz (0.0–1.0) |

**Antwort:**
```json
{
  "num_predictions": 3,
  "predictions": [
    {"go_term": "GO:0008150", "confidence": 0.953},
    {"go_term": "GO:0003674", "confidence": 0.797},
    {"go_term": "GO:0005575", "confidence": 0.712}
  ]
}
```

**Fehler-Antworten:**

| Code | Beschreibung |
|---|---|
| 422 | Embedding-Dimension stimmt nicht überein |
| 500 | Modell nicht geladen |

---

#### GET /terms

Gibt alle GO-Terme zurück, auf denen das Modell trainiert wurde.

**Antwort:**
```json
{
  "num_terms": 500,
  "terms": ["GO:0005575", "GO:0008150", ...]
}
```

---

### Beispiel — Python

```python
import json
import urllib.request
import numpy as np

# Echtes Embedding laden
embedding = np.load("data/embeddings/esm2/train_embeddings.npy")[0].tolist()

data = json.dumps({
    "embedding": embedding,
    "threshold": 0.5
}).encode()

req = urllib.request.Request(
    "http://localhost:8000/predict",
    data=data,
    headers={"Content-Type": "application/json"}
)
antwort = urllib.request.urlopen(req)
ergebnis = json.loads(antwort.read())
print(f"{ergebnis['num_predictions']} GO-Terme vorhergesagt")
for pred in ergebnis["predictions"][:5]:
    print(f"  {pred['go_term']}: {pred['confidence']:.3f}")
```

### Interaktive Dokumentation

Im Browser öffnen: `http://localhost:8000/docs`
