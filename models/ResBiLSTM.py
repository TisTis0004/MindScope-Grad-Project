import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    1D Residual Block as defined in Figure 1 of the ResBiLSTM paper.
    Implements the logic: y_l = h(x_l) + F(x_l; theta_l)[cite: 182].
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()

        # Each convolutional layer uses a 5x1 kernel [cite: 301]
        # Padding is set to 2 to maintain temporal dimensionality for stride=1
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=5, stride=stride, padding=2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=5, stride=1, padding=2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut/Skip Connection [cite: 174, 177]
        # If input/output shapes differ, use a 1x1 convolution to match [cite: 180]
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        # Main path: Conv -> BN -> ReLU -> Conv -> BN [cite: 176]
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Addition before the final ReLU activation [cite: 177]
        out += residual
        return self.relu(out)


class ResBiLSTM(nn.Module):
    """
    Hybrid Deep Learning approach for detecting epileptic seizures.
    Architecture: 1D-ResNet (Spatial) -> BiLSTM (Temporal) -> FC Layers (Classifier)[cite: 35, 37, 38].
    This implementation uses 5 Residual Blocks for optimal binary performance[cite: 391].
    """

    def __init__(self, n_channels: int = 20, n_outputs: int = 2):
        super(ResBiLSTM, self).__init__()

        # 1. SPATIAL FEATURE EXTRACTION (ResNet)
        # Block 1: 64 kernels, stride 2 [cite: 301]
        self.block1 = ResidualBlock(n_channels, 64, stride=2)
        self.drop1 = nn.Dropout(
            0.2
        )  # Dropout rate of 0.2 used after blocks [cite: 299]

        # Block 2: 64 kernels, stride 1 [cite: 302]
        self.block2 = ResidualBlock(64, 64, stride=1)
        self.drop2 = nn.Dropout(0.2)

        # Block 3: 128 kernels, stride 2 [cite: 312, 389]
        self.block3 = ResidualBlock(64, 128, stride=2)
        self.drop3 = nn.Dropout(0.2)

        # Block 4: 128 kernels, stride 1 [cite: 389]
        self.block4 = ResidualBlock(128, 128, stride=1)
        self.drop4 = nn.Dropout(0.2)

        # Block 5: 256 kernels, stride 2 [cite: 389]
        self.block5 = ResidualBlock(128, 256, stride=2)
        self.drop5 = nn.Dropout(0.2)

        # 2. TEMPORAL DEPENDENCY MODELING (BiLSTM)
        # Hidden units set to 64 [cite: 304]
        # Input to LSTM must be [Batch, Seq, Features]
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_drop = nn.Dropout(0.2)

        # 3. CLASSIFIER MODULE [cite: 291]
        # FC1 integrates 128 neurons [cite: 313]
        self.fc_drop1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64 * 2, 128)  # 64*2 because bidirectional
        self.relu_fc = nn.ReLU(inplace=True)

        # FC2 matches number of output classes [cite: 314]
        self.fc_drop2 = nn.Dropout(0.5)  # Final dropout is 0.5 [cite: 315]
        self.fc2 = nn.Linear(128, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape expected: [Batch, Channels, Time(512)] [cite: 298]

        # Spatial Feature Extraction
        x = self.drop1(self.block1(x))
        x = self.drop2(self.block2(x))
        x = self.drop3(self.block3(x))
        x = self.drop4(self.block4(x))
        x = self.drop5(self.block5(x))

        # Prepare for LSTM: Permute to [Batch, Seq_Len, Feature_Dim]
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)

        # Capture contextual output from the last hidden state
        x = x[:, -1, :]

        # Classification
        x = self.lstm_drop(x)
        x = self.relu_fc(self.fc1(self.fc_drop1(x)))
        logits = self.fc2(self.fc_drop2(x))

        return logits
