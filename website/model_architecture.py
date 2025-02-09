import torch
import torch.nn as nn
import torch.nn.functional as F

def weighted_mse_loss(pred, target, weight=10.0):
    pred = pred.squeeze(-1)
    diff = pred - target
    weights = torch.where(target > 0.5, weight, 1.0)
    return torch.mean(weights * diff ** 2)

class HybridZulfModel(nn.Module):
    def __init__(self,
                 input_channels=2,
                 conv_filters=64,
                 lstm_units=128,
                 num_layers=2,
                 dropout=0.2,
                 seq_length=1024):
        super().__init__()
        self.seq_length = seq_length

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=conv_filters,
                      kernel_size=7, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(conv_filters),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=conv_filters, out_channels=conv_filters * 2,
                      kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(conv_filters * 2),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(input_size=conv_filters * 2,
                            hidden_size=lstm_units,
                            num_layers=num_layers,
                            bidirectional=True,
                            dropout=dropout,
                            batch_first=True)

        self.attention = nn.Sequential(
            nn.Linear(2 * lstm_units, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * lstm_units, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
    
        x = self.conv_block(x)       
        x = x.permute(0, 2, 1)         
        lstm_out, _ = self.lstm(x)     
        attn_weights = self.attention(lstm_out)  
        context = torch.sum(attn_weights * lstm_out, dim=1, keepdim=True)  
        enhanced = lstm_out + context.expand_as(lstm_out)
        output = self.fc(enhanced)     
        return output
