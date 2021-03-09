#encoding: utf-8
import os
import glob
import random
from collections import namedtuple


AudioVisualEntry = namedtuple('AudioVisualEntry', ['speaker_id', 'audio_path', 'video_path'])

class AudioVisualDataset:
    def __init__(self, base_path):
        self._base_path = base_path

    def subset(self, speaker_ids, max_files=None, shuffle=False):
        entries = []
        for speaker_id in speaker_ids:
            audio_paths = glob.glob(os.path.join(self._base_path, speaker_id, 'audio', '*.wav'))
            for audio_path in audio_paths:
                entry = AudioVisualEntry(speaker_id, audio_path,
                                         AudioVisualDataset.__audio_to_video_path(audio_path))
                entries.append(entry)
        if shuffle:
            random.shuffle(entries)
        return entries[:max_files]

    def list_speakers(self):
        return os.listdir(self._base_path)

    @staticmethod
    def __audio_to_video_path(audio_path):
        path_names = audio_path.split('\\')
        if path_names[-2] != 'audio':
            raise Exception('invalid audio-video path conversion')

        path_names[-2] = 'video'
        return glob.glob(os.path.splitext('\\'.join(path_names))[0] + ".*")[0]


class AudioDataset:
    def __init__(self, base_paths):
        self._base_paths = base_paths

    def subset(self, max_files=None, shuffle=False):
        audio_file_paths = [os.path.join(d, f) for d in self._base_paths for f in os.listdir(d)]
        if shuffle:
            random.shuffle(audio_file_paths)
        return audio_file_paths[:max_files]


def list_speakers(args):
    if args.speakers is None:
        dataset = AudioVisualDataset(args.dataset_dir)
        speaker_ids = dataset.list_speakers()
    else:
        speaker_ids = args.speakers

    if args.ignored_speakers is not None:
        for speaker_id in args.ignored_speakers:
            speaker_ids.remove(speaker_id)

    return speaker_ids


def list_data(dataset_dir, speaker_ids, noise_dirs, max_files=None, shuffle=True,
              augmentation_factor=1):
    speech_dataset = AudioVisualDataset(dataset_dir)
    speech_subset = speech_dataset.subset(speaker_ids, max_files, shuffle)
    
    noise_dataset = AudioDataset(noise_dirs)
    noise_file_paths = noise_dataset.subset(max_files, shuffle)
    
    n_files = len(speech_subset)
    
    while(n_files > len(noise_file_paths)):
        noise_file_paths += noise_dataset.subset(max_files, shuffle)

    speech_entries = speech_subset[:n_files]
    noise_file_paths = noise_file_paths[:n_files]
    
    all_speech_entries = speech_entries
    all_noise_file_paths = noise_file_paths
    
    for i in range(augmentation_factor - 1):
        all_speech_entries += speech_entries
        all_noise_file_paths += random.sample(noise_file_paths, len(noise_file_paths))

    return all_speech_entries, all_noise_file_paths