import numpy as np
import os
import matplotlib.pyplot as plt

FILE_SIZE_MB = 100
num_columns = 3
BYTES_PER_MB = 1024 * 1024
NUM_POINTS = int(FILE_SIZE_MB * BYTES_PER_MB / 8 / num_columns)
SEQ_LENGTH = 1000
dt = 1e-5
num_pulses = 20
FID_points = 50000

A_common = 0.889
phi_common = 0.6435
high_T2, high_f = 0.300, 720
ultra_T2, ultra_f = 0.500, 10

def simulate_complex_FID(t_segment, A, T2, f, phi):
    return A * np.exp(-t_segment / T2) * np.exp(1j * (2 * np.pi * f * t_segment + phi))

def generate_synthetic_nmr_file(field_type, file_index, output_folder):
    t_total = np.arange(NUM_POINTS) * dt
    X = np.zeros(NUM_POINTS)
    Y = np.zeros(NUM_POINTS)
    
    T2 = high_T2 if field_type == "high" else ultra_T2
    f = high_f if field_type == "high" else ultra_f
    
    for i in range(num_pulses):
        start_idx = i * (NUM_POINTS // num_pulses)
        end_idx = start_idx + FID_points
        if end_idx > NUM_POINTS:
            end_idx = NUM_POINTS
        t_segment = np.arange(end_idx - start_idx) * dt
        pulse = simulate_complex_FID(t_segment, A_common, T2, f, phi_common)
        X[start_idx:end_idx] += np.real(pulse)
        Y[start_idx:end_idx] += np.imag(pulse)
    
    noise_level = 0.01
    X += np.random.normal(0, noise_level, NUM_POINTS)
    Y += np.random.normal(0, noise_level, NUM_POINTS)
    
    data = np.column_stack((t_total, X, Y))
    file_name = f"synthetic_{field_type}_{file_index}.npy"
    file_path = os.path.join(output_folder, file_name)
    np.save(file_path, data)
    
    fig = plt.figure()
    plt.plot(t_total, X, label='X')
    plt.plot(t_total, Y, label='Y', alpha=0.7)
    plt.title(f"Synthetic NMR Data: {field_type} Field (File {file_index})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    out_fig = os.path.join(output_folder, f"synthetic_{field_type}_{file_index}.png")
    plt.savefig(out_fig)
    plt.close(fig)

    print(f"Generated {file_path} | Shape: {data.shape}")
    print(f"Saved plot to {out_fig}")

def preprocess_data(base_folder):
    ultra_files = sorted([f for f in os.listdir(base_folder) if "ultra" in f and f.endswith('.npy')])
    high_files = sorted([f for f in os.listdir(base_folder) if "high" in f and f.endswith('.npy')])
    
    X_seqs, y_seqs = [], []
    for u_file, h_file in zip(ultra_files, high_files):
        ultra_data = np.load(os.path.join(base_folder, u_file))
        high_data = np.load(os.path.join(base_folder, h_file))
        
        X = ultra_data[:, 1:3].astype(np.float32)
        y = high_data[:, 1:2].astype(np.float32)
        
        num_sequences = X.shape[0] // SEQ_LENGTH
        X_trunc = X[:num_sequences * SEQ_LENGTH].reshape((-1, SEQ_LENGTH, 2))
        y_trunc = y[:num_sequences * SEQ_LENGTH].reshape((-1, SEQ_LENGTH, 1))
        
        X_seqs.append(X_trunc)
        y_seqs.append(y_trunc)
        print(f"Processed {u_file} | Sequences: {X_trunc.shape[0]}")

    X_all = np.concatenate(X_seqs, axis=0)
    y_all = np.concatenate(y_seqs, axis=0)
    
    np.random.seed(42)
    indices = np.random.permutation(len(X_all))
    split = int(0.8 * len(indices))
    
    processed_dir = os.path.join(base_folder, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    np.save(os.path.join(processed_dir, 'X_train.npy'), X_all[indices[:split]])
    np.save(os.path.join(processed_dir, 'y_train.npy'), y_all[indices[:split]])
    np.save(os.path.join(processed_dir, 'X_test.npy'), X_all[indices[split:]])
    np.save(os.path.join(processed_dir, 'y_test.npy'), y_all[indices[split:]])
    
    print(f"\nPreprocessed data saved to {processed_dir}")
    print(f"Training samples: {split}, Test samples: {len(indices) - split}")

def main():
    base_folder = "synthetic_data"
    os.makedirs(base_folder, exist_ok=True)
    
    print("Generating synthetic ultra-low-field files...")
    for i in range(5):
        generate_synthetic_nmr_file("ultra", i, base_folder)
    
    print("\nGenerating synthetic high-field files...")
    for i in range(5):
        generate_synthetic_nmr_file("high", i, base_folder)
    
    print("\nPreprocessing data...")
    preprocess_data(base_folder)

if __name__ == "__main__":
    main()
