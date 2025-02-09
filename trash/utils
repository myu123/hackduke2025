import numpy as np

def simulate_zulf_spectrum(j_couplings, num_points=1000, linewidth=0.1):

    frequency = np.linspace(-10, 10, num_points)   
    spectra = np.zeros((j_couplings.shape[0], num_points))   

    for i, j in enumerate(j_couplings):
        spectrum = np.zeros_like(frequency)
        for coupling in j:
            spectrum += np.exp(-((frequency - coupling) ** 2) / (2 * linewidth ** 2))
            spectrum += np.exp(-((frequency + coupling) ** 2) / (2 * linewidth ** 2))
        spectra[i] = spectrum

    return frequency, spectra

def reshape_data_for_rnn(X, y, freq_points=1000):
    
    num_samples = X.shape[0]

    y_rnn = y.reshape(num_samples, freq_points, 1)

    X_repeated = np.repeat(X[:, np.newaxis, :], freq_points, axis=1)

    freq_indices = np.linspace(-10, 10, freq_points)
    freq_indices = np.tile(freq_indices, (num_samples, 1))  
    freq_indices = freq_indices[:, :, np.newaxis]   

    X_rnn = np.concatenate([X_repeated, freq_indices], axis=2)
     
    return X_rnn, y_rnn
