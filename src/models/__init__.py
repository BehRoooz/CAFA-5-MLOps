from src.models.mlp import MultiLayerPerceptron
from src.models.cnn1d import CNN1D


def build_model(config) -> "torch.nn.Module":
    """Factory: instantiate the model specified in config.model.type."""
    import torch.nn as nn

    model_type = config.model["type"].lower()
    embedding_dim = config.embedding_dim
    num_labels = config.data["num_labels"]

    if model_type == "mlp":
        hidden_dims = config.model.get("mlp_hidden_dims", [864, 712])
        return MultiLayerPerceptron(
            input_dim=embedding_dim,
            hidden_dims=hidden_dims,
            num_classes=num_labels,
        )
    elif model_type == "cnn1d":
        out_channels = config.model.get("cnn_out_channels", [3, 8])
        kernel_size = config.model.get("cnn_kernel_size", 3)
        return CNN1D(
            input_dim=embedding_dim,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_classes=num_labels,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'mlp' or 'cnn1d'.")
