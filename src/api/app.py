"""FastAPI inference API for CAFA-5 protein function prediction."""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.config import load_config
from src.inference.predictor import load_checkpoint
from src.utils import get_device, setup_logger

setup_logger("cafa5")
logger = logging.getLogger("cafa5")
config = load_config("configs/config.yaml")
device = get_device()
checkpoint_path = config.output_dir / "checkpoints" / "best_model.pt"
if not checkpoint_path.exists():
    raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")
model = load_checkpoint(config, checkpoint_path)
model.eval()
term_names = np.load(
    config.output_dir / f"label_matrix_top{config.num_labels}" / "term_names.npy",
    allow_pickle=True,
)
logger.info("API ready — model loaded from %s", checkpoint_path)

app = FastAPI(
    title="CAFA-5 Protein Function Prediction API",
    description="Predicts GO terms from protein embeddings.",
    version="1.0.0",
)

class PredictRequest(BaseModel):
    embedding: list[float] = Field(..., description="Protein embedding vector")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class GOTermPrediction(BaseModel):
    go_term: str
    confidence: float

class PredictResponse(BaseModel):
    num_predictions: int
    predictions: list[GOTermPrediction]

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": config.model.get("type"),
        "embedding_dim": config.embedding_dim,
        "num_labels": config.num_labels,
        "device": str(device),
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    emb = np.array(request.embedding, dtype=np.float32)
    if emb.shape[0] != config.embedding_dim:
        raise HTTPException(
            status_code=422,
            detail=f"Embedding must be {config.embedding_dim}-dimensional, got {emb.shape[0]}",
        )
    with torch.no_grad():
        tensor = torch.tensor(emb).unsqueeze(0).to(device)
        probs = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
    mask = probs >= request.threshold
    results = [
        GOTermPrediction(go_term=str(term_names[i]), confidence=float(probs[i]))
        for i in np.where(mask)[0]
    ]
    results.sort(key=lambda x: x.confidence, reverse=True)
    return PredictResponse(num_predictions=len(results), predictions=results)

@app.get("/terms")
def list_terms() -> dict:
    return {"num_terms": len(term_names), "terms": term_names.tolist()}
