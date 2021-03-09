# coding: utf-8
import numpy as np
import librosa

from mediaio.audio_io import AudioSignal

def spectral_subtract(mixed_signal, n_fft, hop_length, alpha=1.0):
    #alpha = 2
    #n_fft = 640
    #hop_length = 160

    mixed_signal = mixed_signal.get_data(channel_index=0)
    noise_signal = mixed_signal[:1600]

    D = librosa.core.stft(mixed_signal.astype(np.float64), n_fft=n_fft, hop_length=hop_length)
    sample_magnitude, sample_phase = librosa.core.magphase(D)

    D = librosa.core.stft(noise_signal.astype(np.float64), n_fft=n_fft, hop_length=hop_length)
    noise_magnitude, noise_phase = librosa.core.magphase(D)
    mean_magnitude = np.mean(noise_magnitude, axis=1)
    mean_magnitude = alpha * mean_magnitude
    ss_magnitude = sample_magnitude - mean_magnitude.reshape((mean_magnitude.shape[0],1))
    ss_magnitude = np.clip(ss_magnitude, 0, None)

    reconstructed = librosa.istft(ss_magnitude * sample_phase, hop_length=hop_length)
    reconstructed = AudioSignal(reconstructed, 16000)

    return reconstructed