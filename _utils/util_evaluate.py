#coding: utf-8
import math
import pickle
import numpy as np
from pypesq import pypesq
from collections import namedtuple

import _utils.data_processor


Sample = namedtuple('Sample', [
    'speaker_id',
    'video_file_path',
    'speech_file_path',
    'noise_file_path',
    'video_samples',
    'mixed_spectrograms',
    'speech_spectrograms',
    'noise_spectrograms',
    'mixed_signal',
    'video_frame_rate'
])


def preprocess_sample(path_audio, path_video, path_noise, slice_duration_ms=200):
    video_samples, video_frame_rate = _utils.data_processor.preprocess_video_sample(path_video, slice_duration_ms)
    mixed_spectrograms, speech_spectrograms, noise_spectrograms, mixed_signal = \
        _utils.data_processor.preprocess_audio_pair(
            path_audio, path_noise, slice_duration_ms, video_samples.shape[0],
            video_frame_rate)
    n_slices = min(video_samples.shape[0], mixed_spectrograms.shape[0])
    sample = Sample(
            speaker_id=None,
            video_file_path=path_video,
            speech_file_path=path_audio,
            noise_file_path=path_noise,
            video_samples=video_samples[:n_slices],
            mixed_spectrograms=mixed_spectrograms[:n_slices],
            speech_spectrograms=speech_spectrograms[:n_slices],
            noise_spectrograms=noise_spectrograms[:n_slices],
            mixed_signal=mixed_signal,
            video_frame_rate=video_frame_rate
        )
    return sample


def make_normalizer(args, assets, name_model):
    with open(assets.get_normalization_cache_path(name_model, args.data_dir), 'rb') as normalization_fd:
        video_normalizer = pickle.load(normalization_fd)
    return video_normalizer


"""
def calcurate_snr(signal, noise):
    s = signal.get_data()
    n = noise.get_data()
    snr = 10 * math.log10(np.var(s) / np.var(n))
    return snr
"""

def calcurate_snr(signal, mixed_signal):
    signal = signal.get_data()
    mixed_signal = mixed_signal.get_data()
    n_len = min(signal.size, mixed_signal.size)
    signal = signal[:n_len]
    mixed_signal = mixed_signal[:n_len]
    noise = mixed_signal - signal
    snr = 10 * math.log10(np.var(signal) / np.var(noise))
    return snr


def calcurate_pesq(signal, noise):
    s = signal.get_data()
    n = noise.get_data()
    sr = signal.get_sample_rate()
    n_len = min(s.size, n.size)
    pesq = pypesq(sr, s[:n_len], n[:n_len], 'wb')
    return pesq
