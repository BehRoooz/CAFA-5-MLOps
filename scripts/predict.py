#!/usr/bin/env python
"""CLI: Run inference and produce a CAFA-5 submission file.

Usage:
    python scripts/predict.py --config configs/config.yaml [--checkpoint outputs/checkpoints/best_model.pt]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.inference.predictor import load_checkpoint, predict, save_submission
from src.utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="CAFA-5 prediction / submission")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: outputs/checkpoints/best_model.pt)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logger("cafa5", log_dir=config.output_dir)

    checkpoint_path = args.checkpoint or str(
        config.output_dir / "checkpoints" / "best_model.pt"
    )

    model = load_checkpoint(config, checkpoint_path)
    submission_df = predict(config, model)
    save_submission(submission_df, config.output_dir)


if __name__ == "__main__":
    main()
