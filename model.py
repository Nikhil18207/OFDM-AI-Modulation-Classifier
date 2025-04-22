import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridCNNLSTM(nn.Module):
    def __init__(self, num_classes=6, lstm_hidden=128, lstm_layers=2):
        super(HybridCNNLSTM, self).__init__()

        # CNN feature extractor
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # LSTM for temporal modeling
        self.lstm_hidden = lstm_hidden
        self.lstm = nn.LSTM(input_size=32, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, bidirectional=True)

        # Final classification layer
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        # x: [B, 1, F, T]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [B, 16, F/2, T/2]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B, 32, F/4, T/4]
        x = self.dropout(x)

        # Reshape for LSTM: [B, T', Features]
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2)         # [B, T, C, F]
        x = x.mean(dim=-1)               # [B, T, C] → average over freq dimension

        # LSTM
        x, _ = self.lstm(x)              # [B, T, H*2]
        x = x[:, -1, :]                  # Take last time step
        x = self.fc(x)
        return x
