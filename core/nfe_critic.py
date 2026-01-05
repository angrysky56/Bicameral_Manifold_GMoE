
import torch
import torch.nn as nn

class HiguchiFDCritic(nn.Module):
    """
    1D Convolutional Neural Network for estimating Higuchi Fractal Dimension (D_H).

    Trained on synthetic Fractional Brownian Motion (fBm) data.
    Input: (batch, seq_len)
    Output: (batch, 1) -> Estimated D_H
    """
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x is (batch, seq_len). Add channel dim -> (batch, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.conv_blocks(x)
        return self.head(feat).squeeze(-1)
