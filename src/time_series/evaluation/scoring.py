# import dependencies
import torch
import numpy as np
from src.time_series.models.conv_autoencoder import ConvAutoencoder


def compute_scores(model: ConvAutoencoder, x: torch.Tensor, m: torch.Tensor) -> np.ndarray:
    """One anomaly score per window = mean reconstruction error over real (masked-in) points."""
    model.eval()                                  # no dropout/batchnorm training behavior
    with torch.no_grad():                         # no gradients needed, we're just scoring
        recon = model(x, m)
        error = ((recon - x) ** 2) * m            # squared error, zeroed out where mask=0
        scores = error.sum(dim=(1, 2)) / m.sum(dim=(1, 2))   # per-window average over REAL points only
    return scores.numpy()                         # convert to numpy for easy use outside torch


def get_threshold(train_scores: np.ndarray, percentile: float = 95) -> float:
    """Cutoff = a percentile of TRAIN scores — train stands in for 'normal' behavior."""
    return float(np.percentile(train_scores, percentile))


def flag_anomalies(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean array — True where a window's score exceeds the threshold."""
    return scores > threshold