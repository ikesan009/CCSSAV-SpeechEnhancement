#encoding: utf-8
import os
import glob
import pickle
import numpy as np
import _utils.data_processor


def load_pkl_file(fname):
    with open(fname, 'rb') as fd:
	    sample = pickle.load(fd)
    return sample


def load_preprocessed_blobs(folds, args, assets):
    if(folds == 'train'):
        preprocessed_blob_paths = [assets.get_preprocessed_blob_path(d) for d in args.data_train]
    elif(folds == 'val'):
        preprocessed_blob_paths = [assets.get_preprocessed_blob_path(d) for d in args.data_validation]
    elif(folds == 'test'):
        preprocessed_blob_paths = [assets.get_preprocessed_blob_path(d) for d in args.data_test]

    all_samples = []
    for preprocessed_blob_path in preprocessed_blob_paths:
        all_samples += load_preprocessed_blob(preprocessed_blob_path)

    return all_samples


def load_preprocessed_blob(preprocessed_blob_path):
    print("loading preprocessed samples from %s ..." % preprocessed_blob_path)
    path_video_samples = os.path.join(preprocessed_blob_path, 'video_samples', '*.pkl')
    path_video_samples = glob.glob(path_video_samples)
    path_mixed_spectrograms = os.path.join(preprocessed_blob_path, 'mixed_spectrograms', '*.pkl')
    path_mixed_spectrograms = glob.glob(path_mixed_spectrograms)
    path_ssed_spectrograms = os.path.join(preprocessed_blob_path, 'ssed_spectrograms', '*.pkl')
    path_ssed_spectrograms = glob.glob(path_ssed_spectrograms)
    path_speech_spectrograms = os.path.join(preprocessed_blob_path, 'speech_spectrograms', '*.pkl')
    path_speech_spectrograms = glob.glob(path_speech_spectrograms)
    return path_video_samples, path_mixed_spectrograms, path_ssed_spectrograms, path_speech_spectrograms


def make_normalizer(folds, args, assets, path_video_samples):
    if folds == 'train' and args.normalization == 'True':
        video_samples = np.concatenate([load_pkl_file(sample).reshape([1, 128, 128, -1]) \
            for sample in path_video_samples], axis=0)
        video_normalizer = _utils.data_processor.VideoNormalizer(video_samples)
        with open(assets.get_normalization_cache_path(args.model, args.data_dir), 'wb') as normalization_fd:
            pickle.dump(video_normalizer, normalization_fd)
        print('saved video normalizer.')
    else:
        with open(assets.get_normalization_cache_path(args.model, args.data_dir), 'rb') as normalization_fd:
            video_normalizer = pickle.load(normalization_fd)
        print('loaded video normalizer.')
    return video_normalizer


def video_normalize(video_sample, video_normalizer):
    video_normalizer.normalize(video_sample.reshape([1, 128, 128, -1]))
    return video_sample.reshape([128, 128, -1])


class MyDataset():
    def __init__(self, folds, args, assets):
        self.path_video_samples, self.path_mixed_spectrograms, self.path_ssed_spectrograms, self.path_speech_spectrograms = \
            load_preprocessed_blobs(folds, args, assets)
        self.video_normalizer = make_normalizer(folds, args, assets, self.path_video_samples)


    def __getitem__(self, idx):
        return video_normalize(load_pkl_file(self.path_video_samples[idx]), self.video_normalizer), \
            load_pkl_file(self.path_mixed_spectrograms[idx]), \
            load_pkl_file(self.path_ssed_spectrograms[idx]), \
            load_pkl_file(self.path_speech_spectrograms[idx])


    def __len__(self):
        return len(self.path_video_samples)


    def get_shapes(self):
        return list(load_pkl_file(self.path_video_samples[0]).shape), \
            list(load_pkl_file(self.path_mixed_spectrograms[0]).shape)