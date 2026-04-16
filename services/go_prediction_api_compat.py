from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path("/app")
SERVICE_DIR = APP_ROOT / "services" / "go-prediction-api"

if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from model_loader import load_model  # noqa: E402


def load_checkpoint_bundle(checkpoint_path, meta_path):
    return load_model(checkpoint_path, meta_path, device="cpu")
