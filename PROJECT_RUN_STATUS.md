# CAFA-5-MLOps aktueller Stand

## Erfolgreich getestet

- Python venv mit Python 3.12
- CAFA-5 Daten eingebunden
- Preprocess erfolgreich
- Train/Holdout Split erfolgreich
- AMD ROCm Docker funktioniert
- ESM2 Embeddings auf AMD GPU erzeugt
- Training erfolgreich
- Holdout Evaluation erfolgreich
- MLflow Registry funktioniert
- cafa-go-model Version 2 als champion gesetzt
- go-prediction-api lädt Modell aus Registry
- embedding-api-amd macht Sequenz → GO-Term Prediction

## Wichtige Metriken

- Train Embeddings: 142246 x 1280
- Holdout Embeddings: 14225 x 1280
- Holdout F1 micro: 0.33292003015854527
- Holdout BCE Loss: 0.09754859509744815
- Champion Model: cafa-go-model version 2

## Wichtige lokale Pfade

- Repo: /home/scardia/Projekte/CAFA-5-MLOps
- Daten: data/cafa-5-protein-function-prediction
- Embeddings: data/embeddings/hf_esm2
- Checkpoint: outputs/checkpoints/best_model.pt
- MLflow: http://localhost:5000
- GO Prediction API: http://localhost:8001
- Embedding API: http://localhost:8000

## Wichtige Befehle

Health Prediction API:
curl -s http://localhost:8001/health

Health Embedding API:
curl -s http://localhost:8000/api/v1/health

Sequenz zu GO-Terms:
curl -s http://localhost:8000/api/v1/predict-go-from-sequences \
  -H "Content-Type: application/json" \
  -d '{"sequences":[{"id":"TEST001","sequence":"MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANLQDKPEF"}],"top_k":10}' | jq
