import torch
import torch.nn as nn

class HybridZulfModel(nn.Module):
    def __init__(
        self, 
        input_channels=2, 
        conv_filters=64, 
        lstm_units=128,
        num_layers=2, 
        dropout=0.3, 
        seq_length=1000
    ):
        super().__init__()
        self.seq_length = seq_length
        
        self.conv = nn.Conv1d(
            in_channels=input_channels, 
            out_channels=conv_filters, 
            kernel_size=5, 
            padding=2
        )
        
        self.ln_conv = nn.LayerNorm(conv_filters)

        self.activation = nn.GELU()
        self.dropout_conv = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_units,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * lstm_units, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.ln_conv(x)
        x = self.activation(x)
        x = self.dropout_conv(x)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

def weighted_mse_loss(pred, target, weight=10.0):
    diff = pred - target
    weights = torch.where(target > 0.5, weight, 1.0)
    return torch.mean(weights * diff ** 2)
