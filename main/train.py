import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import psutil
import matplotlib.pyplot as plt
import numpy as np
from hf_ulf_synthetic import simulate_random_nmr, simulate_random_zulf, downsample

class SyntheticNMRDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, seq_length):
        self.num_samples = num_samples
        self.seq_length = seq_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        _, high_field_fid, _ = simulate_random_nmr()
        _, zulf_fid, _ = simulate_random_zulf()
        high_field_ds = downsample(high_field_fid, self.seq_length)
        zulf_ds = downsample(zulf_fid, self.seq_length)
        scale = np.mean(np.abs(high_field_ds)) / (np.mean(np.abs(zulf_ds)) + 1e-8)
        baseline = zulf_ds * scale
        target = high_field_ds - baseline
        input_data = np.stack([baseline, zulf_ds], axis=0)
        return (torch.tensor(input_data, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32))

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

def enforce_channels_first(x):
    if x.dim() == 3:
        if x.shape[1] != 2 and x.shape[2] == 2:
            x = x.permute(0, 2, 1)
    return x

def transform(signal):
    fft_vals = np.fft.fft(signal)
    fft_vals = np.abs(fft_vals)
    half = len(fft_vals) // 2
    freq_axis = np.fft.fftfreq(len(signal), d=1)[:half]
    return freq_axis, fft_vals[:half]

class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.reset()

    def reset(self):
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

from model import HybridZulfModel, weighted_mse_loss

def train():
    config = {
        'batch_size': 128,
        'lr': 5e-3,
        'num_epochs': 2,
        'patience': 5,
        'grad_clip': 2.0,
        'full_precision': True,
        'seq_length': 1024,
        'train_samples': 5000,
        'test_samples': 500
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print_memory_usage()
    
    train_dataset = SyntheticNMRDataset(num_samples=config['train_samples'], seq_length=config['seq_length'])
    test_dataset = SyntheticNMRDataset(num_samples=config['test_samples'], seq_length=config['seq_length'])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'],
                                                shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'],
                                               num_workers=2, pin_memory=True)
    
    model = HybridZulfModel(input_channels=2, seq_length=config['seq_length']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=2, verbose=True)
    early_stopping = EarlyStopping(patience=config['patience'])
    scaler = GradScaler(enabled=config['full_precision'])
    
    best_loss = float('inf')
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0.0
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}") as tepoch:
            for x, y in tepoch:
                x = enforce_channels_first(x)
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                with autocast(device_type=device.type, enabled=config['full_precision']):
                    pred = model(x)
                    loss = weighted_mse_loss(pred, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        epoch_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x = enforce_channels_first(x)
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += weighted_mse_loss(pred, y).item()
        val_loss /= len(test_loader)
        
        print(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print_memory_usage()
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_model.pth')
            print(f"Saved new best model with loss {best_loss:.4f}")
        early_stopping(val_loss)
        scheduler.step(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    plot_predictions(model, device, test_loader)
    plot_fourier_predictions(model, device, test_loader)

def plot_predictions(model, device, test_loader):
    model.eval()
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    with torch.no_grad():
        x, y_true = next(iter(test_loader))
        x = enforce_channels_first(x)
        x, y_true = x[:3].to(device), y_true[:3].to(device)
        y_pred = model(x)
        baseline = x[:, 0, :]
        reconstructed = baseline + y_pred.squeeze(-1)
        plt.figure(figsize=(12, 8))
        for i in range(3):
            plt.subplot(3, 1, i+1)
            true_high_field = baseline[i].cpu().numpy() + y_true[i].cpu().numpy()
            plt.plot(true_high_field, label='True High Field')
            plt.plot(reconstructed[i].cpu().numpy(), label='Reconstructed High Field', alpha=0.7)
            plt.xlabel("Time/Points")
            plt.ylabel("Intensity")
            plt.title(f"Time-Domain Signal (Sample {i+1})")
            plt.legend()
        plt.tight_layout()
        plt.savefig('predictions.png')
        plt.show()

def plot_fourier_predictions(model, device, test_loader):
    model.eval()
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    with torch.no_grad():
        x, y_true = next(iter(test_loader))
        x = enforce_channels_first(x)
        x, y_true = x[:3].to(device), y_true[:3].to(device)
        y_pred = model(x)
        baseline = x[:, 0, :]
        reconstructed = baseline + y_pred.squeeze(-1)
        plt.figure(figsize=(12, 8))
        for i in range(3):
            plt.subplot(3, 1, i+1)
            true_signal = baseline[i].cpu().numpy() + y_true[i].cpu().numpy()
            pred_signal = reconstructed[i].cpu().numpy()
            freq_true, spec_true = transform(true_signal)
            freq_pred, spec_pred = transform(pred_signal)
            plt.plot(freq_true, spec_true, label='True Spectrum')
            plt.plot(freq_pred, spec_pred, label='Reconstructed Spectrum', alpha=0.7)
            plt.xlabel("Frequency")
            plt.ylabel("Intensity")
            plt.title(f"Fourier Transform Spectrum (Sample {i+1})")
            plt.legend()
        plt.tight_layout()
        plt.savefig('fourier_predictions.png')
        plt.show()

if __name__ == "__main__":
    train()
