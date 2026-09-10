"""Generate the simulated sleep EEG recording used in Lab 2.

The simulation combines background EEG, a slow oscillation, measurement noise, sleep spindles, gradual amplitude drift, and two large movement artifacts. It also returns the known spindle and artifact times so the notebook can compare detector output with ground truth.
"""

import numpy as np
from scipy import signal


def simulate_sleep_eeg(fs=100, duration_s=240, seed=7):
    """Simulate a single-channel sleep EEG recording.

    Parameters
    ----------
    fs : int or float
        Sampling rate in hertz.
    duration_s : int or float
        Recording duration in seconds.
    seed : int
        Seed used to make the random simulation reproducible.

    Returns
    -------
    fs : int or float
        Sampling rate in hertz.
    time_s : ndarray, shape (samples,)
        Time of each EEG sample in seconds.
    eeg_uv : ndarray, shape (samples,)
        Simulated EEG voltage in microvolts.
    spindle_mask : ndarray of bool, shape (samples,)
        Ground-truth mask that is True during simulated spindles.
    events : ndarray, shape (spindles, 2)
        Ground-truth spindle onset and offset times in seconds.
    drift : ndarray, shape (samples,)
        Multiplicative gain applied across the recording.
    artifact_intervals : ndarray, shape (artifacts, 2)
        Ground-truth artifact start and stop times in seconds.

    Notes
    -----
    This is a teaching simulation rather than a physiological model. Several
    effects are deliberately exaggerated so their consequences are visible in the short duration we are looking at.
    """
    rng = np.random.default_rng(seed)
    time_s = np.arange(0, duration_s, 1/fs)
    n = time_s.size

    # Create 1/f-like background activity. Scaling Fourier amplitudes by 1/sqrt(frequency) produces power that falls approximately as 1/f.
    freqs = np.fft.rfftfreq(n, d=1/fs)
    spectrum = np.zeros(freqs.size, dtype=complex)
    spectrum[1:] = np.exp(1j*rng.uniform(0,2*np.pi,freqs.size-1))/np.sqrt(freqs[1:])
    background = np.fft.irfft(spectrum, n=n)
    background = 7*background/np.std(background)

    # Gradually reduce the amplitude to represent a nonstationary recording. The same drift affects the background, slow wave, noise, and spindles.
    drift = 1.20 - 0.85*time_s/duration_s
    eeg_uv = drift*(background + 8*np.sin(2*np.pi*0.8*time_s+0.4) + rng.normal(0,2,n))

    # Add spindle bursts at roughly eight-second intervals. Random jitter, duration, frequency, phase, and amplitude keep the events from being identical. The Hann envelope gives each burst a waxing-and-waning shape.
    spindle_mask = np.zeros(n, dtype=bool)
    events = []
    onsets = np.arange(8,duration_s-3,8) + rng.uniform(-2,2,len(np.arange(8,duration_s-3,8)))
    for onset_s in onsets:
        duration = rng.uniform(0.65,1.45)
        start, stop = int(onset_s*fs), min(n,int((onset_s+duration)*fs))
        event_time = np.arange(stop-start)/fs
        spindle_amplitude_uv = rng.uniform(11,16)
        # Keep one post-artifact spindle near the two detectors' decision
        # boundaries so you can inspect their different behavior.
        if 173 < onset_s < 176:
            spindle_amplitude_uv *= 1.04
        burst = spindle_amplitude_uv*np.hanning(stop-start)
        burst *= np.sin(2*np.pi*rng.uniform(10.5,12.8)*event_time+rng.uniform(0,2*np.pi))
        eeg_uv[start:stop] += drift[start:stop]*burst
        spindle_mask[start:stop] = True
        events.append((start/fs,stop/fs))

    # Add two exaggerated movement/electrode artifacts. Each contains both broadband noise and 11.5 Hz energy, so a 10-13 Hz filter cannot remove it completely. A Tukey window smooths the artifact onset and offset.
    artifact_intervals = []
    artifact_specs = ((152,8.5,380),(207,3.0,220))
    for center_s,artifact_duration_s,amplitude_uv in artifact_specs:
        start = int((center_s-artifact_duration_s/2)*fs)
        stop = int((center_s+artifact_duration_s/2)*fs)
        artifact_time = np.arange(stop-start)/fs
        envelope = signal.windows.tukey(stop-start,alpha=0.5)
        broadband = rng.normal(0,1,stop-start)
        spindle_band_contamination = np.sin(2*np.pi*11.5*artifact_time)
        eeg_uv[start:stop] += amplitude_uv*envelope*(0.65*broadband+0.75*spindle_band_contamination)
        artifact_intervals.append((start/fs,stop/fs))
    return fs,time_s,eeg_uv,spindle_mask,np.asarray(events),drift,np.asarray(artifact_intervals)
