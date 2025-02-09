import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.special import comb
import psutil

from model_architecture import HybridZulfModel, weighted_mse_loss, mae_loss


def feature(shift, n, coupling, intensity, x):
    result = np.zeros_like(x, dtype=float)
    for k in range(n + 1):
        result += comb(n, k) * intensity / (n + 1) * np.cos(x * (shift + coupling * (k - n/2)))
    return result

def makefid(feature_values, broadening, x):
    return np.exp(-broadening * x) * feature_values

def transform(fid, resolution=4096):
    fft_vals = np.fft.fft(fid)
    fft_vals = np.real(fft_vals)
    half = len(fft_vals) // 2
    fft_vals = fft_vals[:half]
    freq_axis = np.arange(1, half + 1) / (resolution / (2 * np.pi))
    return freq_axis, fft_vals

def simulate_random_nmr():
    resolution = 4096
    dx = 0.1
    x_vals = np.arange(0, resolution + dx, dx)
    
    shift_a = np.random.uniform(1.0, 1.5)
    n_a = np.random.choice([1, 2, 3])
    coupling_a = np.random.uniform(0.05, 0.15)
    intensity_a = np.random.uniform(2, 4)
    a = feature(shift_a, n_a, coupling_a, intensity_a, x_vals)
    
    shift_b = np.random.uniform(1.8, 2.2)
    n_b = 0
    coupling_b = np.random.uniform(0.05, 0.15)
    intensity_b = np.random.uniform(2, 4)
    b = feature(shift_b, n_b, coupling_b, intensity_b, x_vals)
    
    shift_c = np.random.uniform(4.0, 4.2)
    n_c = np.random.choice([2, 3, 4])
    coupling_c = np.random.uniform(0.05, 0.15)
    intensity_c = np.random.uniform(1, 3)
    c = feature(shift_c, n_c, coupling_c, intensity_c, x_vals)
    
    summed_features = a + b + c
    broadening = np.random.uniform(0.005, 0.02)
    fid = makefid(summed_features, broadening, x_vals)
    spectrum = transform(fid, resolution=resolution)
    return x_vals, fid, spectrum

def simulate_random_zulf():
    resolution = 4096
    dt = 1e-4
    t_vals = np.arange(0, resolution * dt, dt)
    
    peak1_freq = np.random.uniform(110, 115)
    peak2_freq = np.random.uniform(185, 190)
    
    T2 = np.random.uniform(0.9, 1.1)
    
    fid = (np.cos(2 * np.pi * peak1_freq * t_vals) +
           np.cos(2 * np.pi * peak2_freq * t_vals)) * np.exp(-t_vals / T2)
    
    fft_vals = np.fft.fft(fid)
    fft_vals = np.fft.fftshift(fft_vals)
    freqs = np.fft.fftfreq(len(fid), d=dt)
    freqs = np.fft.fftshift(freqs)
    spectrum = (freqs, np.abs(fft_vals))
    return t_vals, fid, spectrum

def downsample(signal, target_length):
    original_length = len(signal)
    indices = np.linspace(0, original_length - 1, target_length)
    return np.interp(indices, np.arange(original_length), signal)


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
        
        mean_val = np.mean(input_data)
        std_val = np.std(input_data) + 1e-8
        input_data = (input_data - mean_val) / std_val
        
        target = (target - mean_val) / std_val
        
        return (
            torch.tensor(input_data, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )


class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
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


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

def enforce_channels_first(x):
    if x.dim() == 3:
        if x.shape[1] != 2 and x.shape[2] == 2:
            x = x.permute(0, 2, 1)
    return x


def train():
    config = {
        'batch_size': 64,
        'lr': 3e-4,
        'num_epochs': 50,
        'patience': 15,
        'grad_clip': 1.0,
        'mixed_precision': True,
        'seq_length': 1000,
        'train_samples': 1000,
        'test_samples': 200
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print_memory_usage()
    
    train_dataset = SyntheticNMRDataset(num_samples=config['train_samples'], seq_length=config['seq_length'])
    test_dataset = SyntheticNMRDataset(num_samples=config['test_samples'], seq_length=config['seq_length'])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=2, pin_memory=True)
    
    model = HybridZulfModel(input_channels=2, seq_length=config['seq_length']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=2, verbose=True)
    
    early_stopping = EarlyStopping(patience=config['patience'])
    scaler = GradScaler(enabled=config['mixed_precision'])
    
    best_loss = float('inf')
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}") as tepoch:
            for x, y in tepoch:
                x = enforce_channels_first(x)
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                
                with autocast(device_type=device.type, enabled=config['mixed_precision']):
                    pred = model(x)
                    loss = mae_loss(pred, y)
                
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
                val_loss += mae_loss(pred, y).item()
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

def plot_predictions(model, device, test_loader):
    
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    x_batch, y_batch = next(iter(test_loader))
    x_batch = enforce_channels_first(x_batch)
    
   
    x_plot = x_batch[:3].to(device)
    y_true_plot = y_batch[:3].to(device)
    
    with torch.no_grad():
        y_pred_plot = model(x_plot)
    

    x_plot_np = x_plot.cpu().numpy()
    y_true_np = y_true_plot.cpu().numpy()
    y_pred_np = y_pred_plot.cpu().numpy()
  
    baseline = x_plot_np[:, 0, :]
    
    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.subplot(3, 1, i+1)
       
        true_high_field = baseline[i] + y_true_np[i]
        reconstructed = baseline[i] + y_pred_np[i]
        
        plt.plot(true_high_field, label='True High Field')
        plt.plot(reconstructed, label='Reconstructed High Field', alpha=0.7)
        plt.legend()
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()

if __name__ == "__main__":
    train()
