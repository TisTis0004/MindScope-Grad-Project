import torch
import torch.nn as nn
import torchvision.models as models

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=7, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size=kernel_size,
            stride=1, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_filters: int = 64,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels, base_filters,
                kernel_size=15, stride=2, padding=7, bias=False
            ),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(
            ResBlock1D(base_filters, base_filters, stride=1, kernel_size=kernel_size, dropout=dropout),
            ResBlock1D(base_filters, base_filters, stride=1, kernel_size=kernel_size, dropout=dropout),
        )

        self.layer2 = nn.Sequential(
            ResBlock1D(base_filters, base_filters * 2, stride=2, kernel_size=kernel_size, dropout=dropout),
            ResBlock1D(base_filters * 2, base_filters * 2, stride=1, kernel_size=kernel_size, dropout=dropout),
        )

        self.layer3 = nn.Sequential(
            ResBlock1D(base_filters * 2, base_filters * 4, stride=2, kernel_size=kernel_size, dropout=dropout),
            ResBlock1D(base_filters * 4, base_filters * 4, stride=1, kernel_size=kernel_size, dropout=dropout),
        )

        self.layer4 = nn.Sequential(
            ResBlock1D(base_filters * 4, base_filters * 8, stride=2, kernel_size=kernel_size, dropout=dropout),
            ResBlock1D(base_filters * 8, base_filters * 8, stride=1, kernel_size=kernel_size, dropout=dropout),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(base_filters * 8, num_classes)

    def forward(self, x):
        # x: [B, C, T]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)   # [B, F]
        x = self.head(x)               # [B, num_classes]
        return x
    
    
    
    



def restnet18_2d(num_channels: int, num_classes: int):
    """
    Modifies ResNet18 to process small [C, 65, 32] EEG spectrograms.
    """
    # 1. Load an untrained ResNet18
    # (ImageNet weights look for visual textures like fur/metal, which doesn't help EEG)
    model = models.resnet18(weights=None)

    # 2. Swap out the first Convolutional Layer
    # - Change input from 3 to 'num_channels'
    # - Shrink the kernel from 7x7 to 3x3
    # - Change stride from 2 to 1 so we don't aggressively shrink your 65x32 image
    model.conv1 = nn.Conv2d(
        in_channels=num_channels, 
        out_channels=64, 
        kernel_size=(3, 3), 
        stride=(1, 1), 
        padding=(1, 1), 
        bias=False
    )

    # 3. Bypass the initial MaxPool layer
    # Your image is already small. If we pool it now, we lose too much temporal detail.
    # We replace it with an Identity layer (which does nothing and just passes the data through).
    model.maxpool = nn.Identity()

    # 4. Swap out the final classification head
    # Point the final fully connected layer to your specific number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model





import torch
import torch.nn as nn

class Spectrogram_CNN_LSTM(nn.Module):
    def __init__(self, num_channels=17, num_classes=2, lstm_hidden=64):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # Collapse frequency to 1, keep time dimension
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        # After pooling: [B, 128, 1, T]
        # So each time step has 128 features
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(lstm_hidden * 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: [B, C, F, T]
        features = self.cnn(x)                  # [B, 128, F', T']
        features = self.adaptive_pool(features) # [B, 128, 1, T']

        features = features.squeeze(2)          # [B, 128, T']
        features = features.permute(0, 2, 1)    # [B, T', 128]

        lstm_out, _ = self.lstm(features)       # [B, T', 2*lstm_hidden]
        last_time_step = lstm_out[:, -1, :]     # [B, 2*lstm_hidden]

        out = self.dropout(last_time_step)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out