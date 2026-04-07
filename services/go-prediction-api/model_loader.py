from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.cnn1d import CNN1D
from src.models.mlp import MultiLayerPerceptron


def load_model_meta(meta_path: str | Path) -> dict[str, Any]:
    path = Path(meta_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_model_from_meta(meta: dict[str, Any]) -> torch.nn.Module:
    model_type = meta["model_type"]
    embedding_dim = int(meta["embedding_dim"])
    num_labels = int(meta["num_labels"])

    if model_type == "cnn1d":
        model = CNN1D(
            input_dim=embedding_dim,
            num_classes=num_labels,
            out_channels=[3, 8],
            kernel_size=3,
        )
    elif model_type == "mlp":
        model = MultiLayerPerceptron(
            input_dim=embedding_dim,
            num_classes=num_labels,
            hidden_dims=[864, 712],
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return model


def load_term_names(term_names_path: str | Path) -> np.ndarray:
    return np.load(Path(term_names_path), allow_pickle=True)


def load_model(
    checkpoint_path: str | Path,
    meta_path: str | Path,
    device: str = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    meta = load_model_meta(meta_path)
    model = build_model_from_meta(meta)

    ckpt = torch.load(Path(checkpoint_path), map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, meta
