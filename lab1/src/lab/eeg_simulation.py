import numpy as np

CHANNEL_NAMES = (
    "Fp1", "Fp2", "F7", "F3", "F4", "F8", "T3", "C3",
    "C4", "T4", "T5", "P3", "P4", "T6", "O1", "O2",
)
ARTIFACT_AMPLITUDES_UV = np.array(
    [5,5,5,5,5,5,5,5,
     5,5,5,5,5,5,5,5]
)
ALPHA_AMPLITUDES_UV = np.array(
    [4, 4, 4, 2, 4, 4, 2, 8, 
     4, 4, 4, 2, 4, 4, 8, 8]
)

def make_time_axis(sample_rate_hz, duration_s, start_s=0.0):
    return np.arange(start_s, start_s + duration_s, 1 / sample_rate_hz)

def simulate_clean_erp(time_s):
    """Return a 16-channel ERP with a focal P300 centered at C3."""
    p300 =   50 * np.exp(-0.5 * ((time_s - 0.300) / 0.050) ** 2)
    p300 += -20 * np.exp(-0.5 * ((time_s - 0.200) / 0.050) ** 2)
    clean_erp = np.zeros((len(CHANNEL_NAMES), time_s.size))
    clean_erp[CHANNEL_NAMES.index("C3")] = p300
    clean_erp[CHANNEL_NAMES.index("F3")] = 0.7 * p300
    clean_erp[CHANNEL_NAMES.index("T3")] = 0.7 * p300
    clean_erp[CHANNEL_NAMES.index("P3")] = 0.7 * p300
    return clean_erp

def simulate_one_over_f_noise(n_samples, sample_rate_hz, rng, scale_uv=2):
    freqs = np.fft.rfftfreq(n_samples, d=1 / sample_rate_hz)
    phases = rng.uniform(0, 2 * np.pi, size=freqs.size)
    amplitude = np.zeros_like(freqs)
    amplitude[1:] = 1 / np.sqrt(freqs[1:])
    spectrum = amplitude * np.exp(1j * phases)
    noise = np.fft.irfft(spectrum, n=n_samples)
    noise = noise / np.std(noise)
    return scale_uv * noise

def johnson_nyquist_rms_uv(resistance_ohm, temperature_k, bandwidth_hz):
    """Johnson-Nyquist RMS voltage, converted from volts to microvolts."""
    boltzmann_j_per_k = 1.380649e-23
    return 1e6 * np.sqrt(
        4 * boltzmann_j_per_k * temperature_k * resistance_ohm * bandwidth_hz
    )

def simulate_alpha(time, A=12, f=8, phase=0.0, onset_s=0.0, pre_onset_scale=0.25):
    alpha = A * np.sin(2 * np.pi * f * time + phase)
    alpha = np.where(time >= onset_s, alpha, pre_onset_scale * alpha)
    return alpha

def simulate_eeg(n_trials=100):
    """Simulate stimulus-locked trials on a 16-channel EEG montage."""
    rng = np.random.default_rng(seed=42)
    sample_rate_hz = 1000
    time = make_time_axis(sample_rate_hz, duration_s=2.0, start_s=-0.2)
    clean_erp = simulate_clean_erp(time)
    n_channels = len(CHANNEL_NAMES)
    n_samples = time.size
    alpha_phase = 0.0
    alpha_phase_jitter = rng.normal(0, 0.35, size=(1, n_channels, 1))
    alpha = simulate_alpha(
        time,
        A=ALPHA_AMPLITUDES_UV[None, :, None],
        phase=alpha_phase + alpha_phase_jitter,
    )

    # A single waveform with a smooth anterior-to-posterior amplitude gradient.
    line_waveform = np.sin(2 * np.pi * 60 * time)
    environment_noise = np.broadcast_to(
        ARTIFACT_AMPLITUDES_UV[None, :, None]
        * line_waveform[None, None, :],
        (n_trials, n_channels, n_samples),
    ).copy()
    common_phase = rng.uniform(0, 2 * np.pi, size=(n_trials, 1, 1))
    common_noise = 8 * np.sin(2 * np.pi * 60 * time + common_phase)
    common_noise = np.broadcast_to(
        common_noise,
        (n_trials, n_channels, n_samples),
    ).copy()

    # Electrode artifact is thermal noise from a 1-Mohm electrode impedance.
    electrode_resistance_ohm = 1e6
    temperature_k = 300
    bandwidth_hz = sample_rate_hz / 2
    electrode_rms_uv = johnson_nyquist_rms_uv(
        electrode_resistance_ohm, temperature_k, bandwidth_hz
    )
    electrode_noise = rng.normal(
        scale=electrode_rms_uv, size=(n_trials, n_channels, n_samples)
    )

    # Trial-varying physiological background; there is no electronics artifact.
    pink = np.empty((n_trials, n_channels, n_samples))
    for i_trial in range(n_trials):
        for i_channel in range(n_channels):
            pink[i_trial, i_channel] = simulate_one_over_f_noise(
                n_samples, sample_rate_hz, rng
            )

    total_noise = electrode_noise + environment_noise + common_noise
    recorded_eeg = clean_erp[None, :, :] + alpha + total_noise + pink

    return (
        sample_rate_hz,
        time,
        clean_erp,
        recorded_eeg,
    )
