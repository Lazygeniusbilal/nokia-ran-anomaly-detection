# import dependencies
import os
import pickle
from pathlib import Path

import torch
from src.time_series.models.conv_autoencoder import ConvAutoencoder


def save_artifacts(model: ConvAutoencoder, scalers: dict, save_dir: str) -> None:
    """Save the trained model's weights and the fitted scalers to disk."""
    os.makedirs(save_dir, exist_ok=True)                          # create folder if it doesn't exist

    model_path = Path(save_dir) / "model.pt"
    torch.save(model.state_dict(), model_path)                    # save only the learned weights, not the whole object

    scalers_path = Path(save_dir) / "scalers.pkl"
    with open(scalers_path, "wb") as f:
        pickle.dump(scalers, f)                                   # sklearn scalers aren't torch objects — use pickle


def load_artifacts(save_dir: str, n_features: int = 5) -> tuple[ConvAutoencoder, dict]:
    """Rebuild the model architecture, load its saved weights, and load the scalers."""
    model = ConvAutoencoder(n_features=n_features)                # fresh, untrained architecture

    model_path = Path(save_dir) / "model.pt"
    model.load_state_dict(torch.load(model_path))                 # pour the saved weights into it
    model.eval()                                                  # switch to inference mode (no training behavior)

    scalers_path = Path(save_dir) / "scalers.pkl"
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    return model, scalers