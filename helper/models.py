import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
try:
    from transformers import ASTForAudioClassification
except ImportError:
    ASTForAudioClassification = None

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

    num_features = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, num_classes)
    )

    return model





import torch
import torch.nn as nn

import torch.nn as nn

class SEBlock2d(nn.Module):
    def __init__(self, channels, reduction=2): 
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class TemporalAttentionPool(nn.Module):
    """
    Attention-based temporal pooling (replaces taking only the last LSTM time step).
    
    WHY: If a seizure starts at t=0 in a 10s window, the LSTM must carry that info
    through 80+ time steps to the end. Attention lets the model look BACK at all
    time steps and weight them by relevance — seizure-onset steps get high attention.
    
    This is the standard approach in papers like:
    - Saab et al. (2020) "Weak supervision as an efficient approach for automated seizure detection"
    - Attention-based temporal models for EEG classification
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False)
        )
    
    def forward(self, lstm_output):
        # lstm_output: [B, T, H]
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)  # [B, T, 1]
        context = (attn_weights * lstm_output).sum(dim=1)  # [B, H]
        return context


class Spectrogram_CNN_LSTM(nn.Module):
    """
    Upgraded CNN-LSTM with:
    1. InstanceNorm2d (domain-invariant, replaces BatchNorm2d)
    2. Temporal Attention Pooling (replaces last-timestep)
    3. Multi-scale convolutions (parallel 3x3 + 5x5 kernels)
    """
    def __init__(self, num_channels=18, num_classes=2, lstm_hidden=64):
        super().__init__()

        # Channel attention (Squeeze-and-Excite)
        self.se = SEBlock2d(num_channels, reduction=2)

        # === MULTI-SCALE CNN with InstanceNorm ===
        # InstanceNorm computes stats PER-SAMPLE, PER-CHANNEL.
        # Unlike BatchNorm (which stores training stats and applies them at eval),
        # InstanceNorm is inherently domain-invariant — it normalizes each sample
        # independently, so a noisy eval recording gets the same treatment as a clean one.
        
        # Block 1: 18 -> 32
        self.conv1_3x3 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=1)
        self.conv1_5x5 = nn.Conv2d(num_channels, 16, kernel_size=5, padding=2)
        self.norm1 = nn.InstanceNorm2d(32, affine=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Block 2: 32 -> 64
        self.conv2_3x3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2_5x5 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Block 3: 64 -> 128
        self.conv3_3x3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_5x5 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.norm3 = nn.InstanceNorm2d(128, affine=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.spatial_dropout = nn.Dropout2d(p=0.3)

        # Freq dimension reduces from 40 -> 20 -> 10 -> 5
        cnn_out_channels = 128
        freq_out = 5
        lstm_input_size = cnn_out_channels * freq_out

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=0.4,
            batch_first=True,
            bidirectional=True
        )

        # === ATTENTION TEMPORAL POOLING ===
        # Instead of lstm_out[:, -1, :], attend over ALL time steps
        self.temporal_attention = TemporalAttentionPool(lstm_hidden * 2)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(lstm_hidden * 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def _cnn_block(self, x, conv_3x3, conv_5x5, norm, pool):
        """Multi-scale convolution: concatenate 3x3 and 5x5 features."""
        out_3 = conv_3x3(x)
        out_5 = conv_5x5(x)
        out = torch.cat([out_3, out_5], dim=1)  # Concat along channel dim
        out = norm(out)
        out = F.relu(out)
        out = pool(out)
        return out

    def forward(self, x):
        # x: [B, C, F, T]
        x = self.se(x)

        # Multi-scale CNN with InstanceNorm
        x = self._cnn_block(x, self.conv1_3x3, self.conv1_5x5, self.norm1, self.pool1)
        x = self._cnn_block(x, self.conv2_3x3, self.conv2_5x5, self.norm2, self.pool2)
        x = self._cnn_block(x, self.conv3_3x3, self.conv3_5x5, self.norm3, self.pool3)
        x = self.spatial_dropout(x)

        # Reshape for LSTM: [B, 128, F', T'] -> [B, T', 128*F']
        B, C, F, T = x.size()
        x = x.view(B, C * F, T)
        x = x.permute(0, 2, 1)

        # Bidirectional LSTM
        lstm_out, _ = self.lstm(x)

        # Attention pooling over all time steps (not just the last one!)
        attended = self.temporal_attention(lstm_out)

        out = self.dropout(attended)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out
    
    
class CNN_LSTM_Large(nn.Module):
    """
    Scaled up version of CNN-LSTM targeting ~2.2 Million parameters.
    Thick CNN channels to extract rich features, deep LSTM to preserve sequence context.
    """
    def __init__(self, num_channels=18, num_classes=2, lstm_hidden=128):
        super().__init__()
        
        # Channel attention
        self.se = SEBlock2d(num_channels, reduction=2)
        
        # Block 1 (18 -> 64)
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Block 2 (64 -> 128)


        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(128, affine=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Block 3 (128 -> 256)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm2d(256, affine=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.spatial_dropout = nn.Dropout2d(p=0.3)
        
        # Freq path: 40 -> 20 -> 10 -> 5
        cnn_out_channels = 256
        freq_out = 5
        lstm_input_size = cnn_out_channels * freq_out  # 1280
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=0.4,
            batch_first=True,
            bidirectional=True
        )
        
        self.temporal_attention = TemporalAttentionPool(lstm_hidden * 2)
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.se(x)
        
        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = self.pool3(F.relu(self.norm3(self.conv3(x))))
        x = self.spatial_dropout(x)
        
        B, C, F_dim, T_dim = x.size()
        x = x.view(B, C * F_dim, T_dim)
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        attended = self.temporal_attention(lstm_out)
        
        out = self.dropout(attended)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out


class ResNet18_LSTM(nn.Module):
    """
    CRNN Hybrid: ResNet18 deep feature extractor + Bidirectional LSTM + Temporal Attention.
    Designed specifically for heavy augmentation robustness.
    """
    def __init__(self, num_channels=18, num_classes=2, lstm_hidden=128):
        super().__init__()
        
        # 1. ResNet18 Deep Feature Extractor
        self.resnet = models.resnet18(weights=None)
        
        # Modify the first conv layer to accept 'num_channels' (18 for EEG spectrograms)
        self.resnet.conv1 = nn.Conv2d(
            in_channels=num_channels, 
            out_channels=64, 
            kernel_size=(3, 3), 
            stride=(1, 1), 
            padding=(1, 1), 
            bias=False
        )
        
        # Bypass initial MaxPool to preserve time resolution for our spectrograms
        self.resnet.maxpool = nn.Identity()
        
        # We manually construct the feature extractor to skip ResNet's AdaptiveAvgPool 
        # (which destroys the 'Time' dimension) and its fully connected layer.
        self.feature_extractor = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool, # Identity
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        )
        
        # 2. LSTM Sequence Modeling
        # Input shape to LSTM: ResNet18 layer4 outputs 512 channels.
        # Original spectrogram freq = 40. 
        # layer1: no downsample
        # layer2: stride 2 -> 40/2 = 20
        # layer3: stride 2 -> 20/2 = 10
        # layer4: stride 2 -> 10/2 = 5
        # So we get 512 channels * 5 frequency bands = 2560 features per time step!
        resnet_out_channels = 512
        freq_out = 5
        lstm_input_size = resnet_out_channels * freq_out
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=0.5,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. Temporal Attention & Classification Head
        self.temporal_attention = TemporalAttentionPool(lstm_hidden * 2)
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: [B, C, F, T] (e.g., [Batch, 18, 40, Time])
        
        # Extract features -> shape: [B, 512, 5, T_out]
        x = self.feature_extractor(x)
        
        # Reshape for LSTM: flatten Channels & Freqs -> [B, 512*5, T_out]
        B, C, F, T = x.size()
        x = x.view(B, C * F, T)
        
        # Swap axes for LSTM: [B, T_out, Features]
        x = x.permute(0, 2, 1)
        
        # Contextual sequence tracking
        lstm_out, _ = self.lstm(x)
        
        # Weighted pooling of the sequence 
        attended = self.temporal_attention(lstm_out)
        
        # Classification
        out = self.dropout(attended)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out
        

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=1, num_heads=1, ff_dim=512, dropout=0.1):
        super().__init__()
        # Layer Normalization [cite: 328]
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Multi-Head Attention [cite: 339]
        # Note: PyTorch requires embed_dim to be divisible by num_heads. 
        # Since input feature dim is 1, we project it to a higher dimension first, 
        # or treat the 1276 features as the sequence.
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, 
                                         dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-Forward Network using 1D Convolutions [cite: 375]
        # input shape for conv1d: (Batch, Channels, Length)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=ff_dim, kernel_size=1)# [cite: 376]
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=embed_dim, kernel_size=1)# [cite: 376]
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1. Layer Norm & MHA
        x_norm = self.norm1(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out) # Residual connection [cite: 344]
        
        # 2. Layer Norm & Conv1D Feed-Forward
        x_norm = self.norm2(x)
        # Reshape for Conv1D: (Batch, Seq_Len, Embed_Dim) -> (Batch, Embed_Dim, Seq_Len)
        x_conv = x_norm.transpose(1, 2)
        x_conv = self.relu(self.conv1(x_conv))
        x_conv = self.conv2(x_conv)
        # Reshape back: (Batch, Embed_Dim, Seq_Len) -> (Batch, Seq_Len, Embed_Dim)
        x_conv = x_conv.transpose(1, 2)
        
        x = x + self.dropout(x_conv) # Residual connection
        return x

class MHACNN(nn.Module):
    def __init__(self, seq_len=1276, num_classes=2, num_blocks=2):
        super().__init__()
        
        # In the paper, input is (M, 1276, 1) [cite: 309]
        # We will project the dimension of 1 to a higher hidden dimension (e.g., 256) 
        # to allow for the 8 attention heads mentioned in Table 3[cite: 323].
        self.hidden_dim = 256 
        self.num_heads = 8    
        self.ff_dim = 512     
        
        # Initial projection to match head dimensions
        self.input_projection = nn.Linear(1, self.hidden_dim)
        
        # Transformer Blocks [cite: 325]
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=self.hidden_dim, 
                num_heads=self.num_heads, 
                ff_dim=self.ff_dim, 
                dropout=0.1
            ) for _ in range(num_blocks)
        ])
        
        # Fully Connected Layers / MLP [cite: 353]
        self.mlp = nn.Sequential(
            nn.Linear(seq_len * self.hidden_dim, 128), # MLP units = 128 [cite: 323]
            nn.ReLU(), #[cite: 354]
            nn.Dropout(0.5), # MLP dropout = 0.5 [cite: 323]
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape expected: (Batch, 1276, 1)
        x = self.input_projection(x) # -> (Batch, 1276, 256)
        
        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Global Average Pooling [cite: 382]
        # The paper pools to (M, 1276), but since we projected to hidden_dim, 
        # we flatten the sequence and project down, or pool across the sequence. 
        # For an exact match to the MLP layer taking high-level features:
        x = torch.flatten(x, start_dim=1) 
        
        # Fully connected & Output [cite: 363]
        logits = self.mlp(x)
        
        # Note: PyTorch's CrossEntropyLoss handles the Softmax internally, 
        # so we output raw logits here.
        return logits

