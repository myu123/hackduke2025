import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

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
    a = feature(shift=shift_a, n=n_a, coupling=coupling_a, intensity=intensity_a, x=x_vals)

    shift_b = np.random.uniform(1.8, 2.2)
    n_b = 0
    coupling_b = np.random.uniform(0.05, 0.15)
    intensity_b = np.random.uniform(2, 4)
    b = feature(shift=shift_b, n=n_b, coupling=coupling_b, intensity=intensity_b, x=x_vals)

    shift_c = np.random.uniform(4.0, 4.2)
    n_c = np.random.choice([2, 3, 4])
    coupling_c = np.random.uniform(0.05, 0.15)
    intensity_c = np.random.uniform(1, 3)
    c = feature(shift=shift_c, n=n_c, coupling=coupling_c, intensity=intensity_c, x=x_vals)

    summed_features = a + b + c

    broadening = np.random.uniform(0.005, 0.02)
    fid = makefid(summed_features, broadening, x_vals)
    spectrum = transform(fid, resolution=resolution)
    return x_vals, fid, spectrum

def simulate_random_zulf():
    resolution = 4096
    dt = 1e-4
    t_vals = np.arange(0, resolution * dt, dt)

    peak1_freq = np.random.uniform(100, 120)
    peak2_freq = np.random.uniform(180, 200)

    T2 = np.random.uniform(0.8, 1.2)
    fid = (np.cos(2 * np.pi * peak1_freq * t_vals) +
           np.cos(2 * np.pi * peak2_freq * t_vals)) * np.exp(-t_vals / T2)

    fft_vals = np.fft.fft(fid)
    fft_vals = np.fft.fftshift(fft_vals)
    freqs = np.fft.fftfreq(len(fid), d=dt)
    freqs = np.fft.fftshift(freqs)
    spectrum = (freqs, np.abs(fft_vals))
    return t_vals, fid, spectrum

def export_csv(data, folder, filename, header=None):
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, filename)
    np.savetxt(filepath, data, delimiter=",", header=header or "", comments="")
    print(f"Saved file: {filepath}")

def main():
    n_simulations = 500
    base_folder = "fid_files"
    high_field_folder = os.path.join(base_folder, "high_field")
    zulf_folder = os.path.join(base_folder, "ultra_low_field")
    
    hf_fid_folder = os.path.join(high_field_folder, "nmr_fid")
    hf_spec_folder = os.path.join(high_field_folder, "nmr_spectrum")
    zulf_fid_folder = os.path.join(zulf_folder, "zulf_fid")
    zulf_spec_folder = os.path.join(zulf_folder, "zulf_spectrum")
    
    for folder in [hf_fid_folder, hf_spec_folder, zulf_fid_folder, zulf_spec_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    for i in range(n_simulations):
        x_nmr, fid_nmr, spectrum_nmr = simulate_random_nmr()
        freq_nmr, spec_nmr = spectrum_nmr

        t_zulf, fid_zulf, spectrum_zulf = simulate_random_zulf()
        freq_zulf, spec_zulf = spectrum_zulf

        fid_filename = f"nmr_fid_{i:04d}.csv"
        spec_filename = f"nmr_spectrum_{i:04d}.csv"
        export_csv(np.column_stack([x_nmr, fid_nmr]), hf_fid_folder, fid_filename, header="Time,Intensity")
        export_csv(np.column_stack([freq_nmr, spec_nmr]), hf_spec_folder, spec_filename, header="Frequency,Intensity")

        fid_filename_z = f"zulf_fid_{i:04d}.csv"
        spec_filename_z = f"zulf_spectrum_{i:04d}.csv"
        export_csv(np.column_stack([t_zulf, fid_zulf]), zulf_fid_folder, fid_filename_z, header="Time,Intensity")
        export_csv(np.column_stack([freq_zulf, spec_zulf]), zulf_spec_folder, spec_filename_z, header="Frequency,Intensity")

        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{n_simulations} simulations.")

    plt.figure(figsize=(8, 6))
    plt.plot(freq_nmr, spec_nmr, lw=1)
    plt.xlim(0, 5)
    plt.xlabel("Frequency (arb. units)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("High Field NMR Spectrum (Sample)")
    sample_hf_plot = os.path.join(high_field_folder, "sample_nmr_spectrum.png")
    plt.savefig(sample_hf_plot)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(freq_zulf, spec_zulf, lw=1)
    plt.xlim(0, 300)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("Simulated ZULF NMR Spectrum (Sample)")
    sample_zulf_plot = os.path.join(zulf_folder, "sample_zulf_spectrum.png")
    plt.savefig(sample_zulf_plot)
    plt.close()

    print("Data generation complete.")

if __name__ == '__main__':
    main()
