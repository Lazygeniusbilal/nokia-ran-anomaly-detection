import torch
import torch.nn as nn
from torch import Tensor


class ConvAutoencoder(nn.Module):
    """
    Mask-aware Conv1D autoencoder for anomaly detection on RAN KPI windows.
    Input: values (batch, window, n_features) + mask (batch, window, n_features),
    concatenated as extra input channels so the model itself can reason about
    what's real vs. filled — not just the loss.
    """

    def __init__(self, n_features: int = 5):
        super().__init__()
        in_ch = n_features * 2  # values + mask concatenated

        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, n_features, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        inp = torch.cat([x, mask], dim=2)   # (batch, window, n_features*2)
        inp = inp.permute(0, 2, 1)          # (batch, channels, window)
        z = self.encoder(inp)
        out = self.decoder(z)
        return out.permute(0, 2, 1)         # (batch, window, n_features)


def masked_mse(recon: Tensor, x: Tensor, mask: Tensor) -> Tensor:
    """Reconstruction error counted only where mask=1 (real data)."""
    error = (recon - x) ** 2
    return (error * mask).sum() / mask.sum()

obj= ConvAutoencoder()