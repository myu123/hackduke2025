import numpy as np
from scipy.special import comb
from scipy.ndimage import gaussian_filter1d

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

def gaussian_smoothing(data, sigma):
    return gaussian_filter1d(data, sigma=sigma)

def simulate_high_field_nmr(smoothing_sigma=4):
    resolution = 4096
    dx = 0.1
    x_vals = np.arange(0, resolution + dx, dx)
    a = feature(shift=1.2, n=2, coupling=0.1, intensity=3, x=x_vals)
    b = feature(shift=2.0, n=0, coupling=0.1, intensity=3, x=x_vals)
    c = feature(shift=4.1, n=3, coupling=0.1, intensity=2, x=x_vals)
    summed_features = a + b + c
    broadening = 0.01
    fid = makefid(summed_features, broadening, x_vals)
    fid_smoothed = gaussian_smoothing(fid, sigma=smoothing_sigma)
    spectrum = transform(fid_smoothed, resolution=resolution)
    return x_vals, fid_smoothed, spectrum

def simulate_zulf_nmr(smoothing_sigma=3):
    resolution = 4096
    dt = 0.1
    t_vals = np.arange(0, resolution + dt, dt)
    a = feature(shift=0, n=2, coupling=0.1, intensity=3, x=t_vals)
    b = feature(shift=0, n=0, coupling=0.1, intensity=3, x=t_vals)
    c = feature(shift=0, n=3, coupling=0.1, intensity=2, x=t_vals)
    summed_features = a + b + c
    broadening = 0.01
    fid = makefid(summed_features, broadening, t_vals)
    fid_smoothed = gaussian_smoothing(fid, sigma=smoothing_sigma)
    spectrum = transform(fid_smoothed, resolution=resolution)
    return t_vals, fid_smoothed, spectrum

def simulate_random_nmr():
    return simulate_high_field_nmr(smoothing_sigma=4)

def simulate_random_zulf():
    return simulate_zulf_nmr(smoothing_sigma=3)

def downsample(signal, target_length):
    original_length = len(signal)
    indices = np.linspace(0, original_length - 1, target_length)
    return np.interp(indices, np.arange(original_length), signal)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    t, fid, spec = simulate_high_field_nmr()
    plt.plot(t, fid)
    plt.title("High Field FID")
    plt.show()
