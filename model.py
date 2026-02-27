import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.
    Ref: Hu et al., 'Squeeze-and-Excitation Networks', CVPR 2018."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class AttentionHybridCNNLSTM(nn.Module):
    """Attention-Enhanced Hybrid CNN-LSTM for OFDM Modulation Classification."""

    def __init__(self, num_classes=6, lstm_hidden=128, lstm_layers=2):
        super().__init__()
        # CNN Stage 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.se1 = SEBlock(16, reduction=4)

        # CNN Stage 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.se2 = SEBlock(32, reduction=4)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size=32, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True,
                            bidirectional=True, dropout=0.2)

        # Classifier
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.se1(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.se2(x)
        x = self.dropout(x)

        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).mean(dim=-1)

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)


# Keep the original for backward compatibility
class HybridCNNLSTM(nn.Module):
    """Original Hybrid CNN-LSTM without attention."""

    def __init__(self, num_classes=6, lstm_hidden=128, lstm_layers=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.lstm_hidden = lstm_hidden
        self.lstm = nn.LSTM(input_size=32, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).mean(dim=-1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)
