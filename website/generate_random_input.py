
import numpy as np


# USERS AND VIEWERS: feel free to use this file to generate random sample input data

def generate_random_fid(num_points=1024, num_peaks=2, max_time=1.0, noise_level=0.02):

    t = np.linspace(0, max_time, num_points)

    real_data = np.zeros(num_points)
    imag_data = np.zeros(num_points)

    for _ in range(num_peaks):
        amplitude = np.random.uniform(0.5, 2.0)
        frequency = np.random.uniform(10, 100)  # in Hz
        phase = np.random.uniform(0, 2 * np.pi)
        damping = np.random.uniform(1, 5)

        real_data += amplitude * np.exp(-damping * t) * np.cos(2 * np.pi * frequency * t + phase)
        imag_data += amplitude * np.exp(-damping * t) * np.sin(2 * np.pi * frequency * t + phase)

    real_data += np.random.normal(scale=noise_level, size=num_points)
    imag_data += np.random.normal(scale=noise_level, size=num_points)

    data = np.column_stack([t, real_data, imag_data])
    return data

def main():
    data = generate_random_fid()

    np.savetxt("random_zulf_fid.txt", data, header="Time Real Imag", comments='')

    print("Generated file 'random_zulf_fid.txt' with random NMR-like fid data.")

if __name__ == "__main__":
    main()