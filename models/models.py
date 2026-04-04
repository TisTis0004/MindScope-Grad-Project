import torch
import torch.nn as nn


class BinarySeizureCNN(nn.Module):
    """Stage 1: High-sensitivity binary seizure detector (~1.08M params)"""

    def __init__(self):
        super().__init__()
        # Input: (Batch, 21 channels, 250 time steps)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(21, 64, kernel_size=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 29, 128),  # 256 * 29 = 7424
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),  # Raw logits for BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return self.classifier(x)


class MultiClassSeizureCNN(nn.Module):
    """Stage 2: Streamlined multiclass type classifier (~1.02M params)"""

    def __init__(self, num_classes=8):
        super().__init__()
        # Input: (Batch, 21 channels, 250 time steps)
        self.conv_block = nn.Sequential(
            nn.Conv1d(21, 128, kernel_size=3), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 124, 64),  # 128 * 124 = 15872
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),  # Raw logits for CrossEntropyLoss
        )

    def forward(self, x):
        x = self.conv_block(x)
        return self.classifier(x)
