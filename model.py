import torch
import torch.nn as nn

class HybridZulfModel(nn.Module):
    def __init__(self, input_channels=2, conv_filters=64, lstm_units=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, conv_filters, 5, padding=2),
            nn.LayerNorm([conv_filters, 1000]),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(conv_filters, lstm_units, num_layers=num_layers,
                           bidirectional=True, dropout=dropout, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(2*lstm_units, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # inp batch, 1000, 2
        x = x.permute(0, 2, 1)  # batch 2 1000
        cnn_out = self.conv(x)
        cnn_out = cnn_out.permute(0, 2, 1)  # bartch 1000 64
        lstm_out, _ = self.lstm(cnn_out)
        return self.fc(lstm_out)

def weighted_mse_loss(pred, target, weight=10.0):
    diff = pred - target
    weights = torch.where(target > 0.5, weight, 1.0)
    return torch.mean(weights * diff ** 2)
