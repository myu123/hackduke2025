import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import HybridZulfModel, weighted_mse_loss
import psutil

class MemoryMappedDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, target_path):
        self.X = np.load(data_path, mmap_mode='r')
        self.y = np.load(target_path, mmap_mode='r')
        assert self.X.shape[0] == self.y.shape[0], "Mismatch in sample count"
        print(f"Loaded dataset: {data_path} | X shape: {self.X.shape}, y shape: {self.y.shape}")

    def __len__(self): return self.X.shape[0]
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
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

def train():
    config = {
        'batch_size': 64,
        'lr': 3e-4,
        'num_epochs': 50,
        'patience': 15,
        'grad_clip': 1.0,
        'mixed_precision': True,
        'seq_length': 1000
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print_memory_usage()
    
    data_dir = "synthetic_data/processed"
    train_dataset = MemoryMappedDataset(
        os.path.join(data_dir, 'X_train.npy'),
        os.path.join(data_dir, 'y_train.npy')
    )
    test_dataset = MemoryMappedDataset(
        os.path.join(data_dir, 'X_test.npy'),
        os.path.join(data_dir, 'y_test.npy')
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'], num_workers=2, pin_memory=True
    )
    
    model = HybridZulfModel(input_channels=2, seq_length=config['seq_length']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=config['patience'])
    scaler = GradScaler(enabled=config['mixed_precision'])
    
    best_loss = float('inf')
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}") as tepoch:
            for x, y in tepoch:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                
                with autocast(enabled=config['mixed_precision']):
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
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += weighted_mse_loss(pred, y).item()
        val_loss /= len(test_loader)
        
        print(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
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
    model.eval()
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    
    with torch.no_grad():
        x, y_true = next(iter(test_loader))
        x, y_true = x[:3].to(device), y_true[:3].to(device)
        y_pred = model(x)
        
        plt.figure(figsize=(12, 8))
        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.plot(y_true[i].cpu().numpy()[:, 0], label='True')
            plt.plot(y_pred[i].cpu().numpy()[:, 0], label='Predicted', alpha=0.7)
            plt.legend()
        plt.tight_layout()
        plt.savefig('predictions.png')
        plt.show()

if __name__ == "__main__":
    train()
