import numpy as np
import matplotlib.pyplot as plt

def sample_from_high_rate(time_high, signal_high, target_rate_hz):
    """Downsample a high-rate signal by selecting the nearest regular samples."""
    sample_times = np.arange(
        time_high[0], time_high[-1] + 1e-12, 1 / target_rate_hz
    )
    indices = np.searchsorted(time_high, sample_times)
    indices = np.clip(indices, 0, signal_high.size - 1)
    return time_high[indices], signal_high[indices]

def rms(values):
    """Root mean square amplitude."""
    return np.sqrt(np.mean(values**2))

def compute_psd(x, fs):
    n = len(x)
    X = np.fft.rfft(x)
    psd = (np.abs(X) ** 2) / (fs * n)
    # Convert two-sided PSD scaling to one-sided PSD
    if n % 2 == 0:
        psd[1:-1] *= 2
    else:
        psd[1:] *= 2
    f = np.fft.rfftfreq(n, d=1/fs)
    return f, psd