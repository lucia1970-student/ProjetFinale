# Feature extraction logic goes here

def extract_features_from_audio(file):
    y, sr = librosa.load(file, sr=None)

    # 1. f0 estimation using librosa.yin
    f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    f0_mean = np.mean(f0)

    # 2. Jitter = relative variation in f0
    jitter = variation(f0)  # standard deviation / mean

    # 3. Shimmer = variation in amplitude envelope
    amplitude_env = np.abs(y)
    shimmer = variation(amplitude_env)

    # 4. HNR (approximate): harmonic-to-noise ratio via energy stats
    signal_energy = np.sum(y ** 2)
    noise_energy = np.sum((y - np.mean(y)) ** 2)
    hnr = 10 * np.log10(signal_energy / (noise_energy + 1e-6))
    hnr = np.clip(hnr, 0, 100)  # avoid wild spikes

    return [f0_mean, jitter, shimmer, hnr]