#coding: utf-8
import os
import sys
import time
import glob
import shutil
import pickle
import logging
import argparse
import multiprocessing
import numpy as np
import librosa
from tqdm import tqdm
from shutil import copy2
from datetime import datetime

import _utils.data_processor
from _utils.util_asset import *
from _utils.util_data import *
from _utils.util_train_test import *
from _utils.util_evaluate import *
from mediaio.audio_io import AudioSignal, AudioMixer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torch.autograd import Variable

from network import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def preprocess(args):
    print('-' * 95)
    print('start preprocessing '+args.data_name+'...\n')
    assets = AssetManager(args)
    speaker_ids = list_speakers(args)

    speech_entries, noise_file_paths = list_data(args.dataset_dir, speaker_ids, 
        args.noise_dirs, max_files=None)

    dir_save = assets.get_preprocessed_blob_path(args.data_name)
    _utils.data_processor.preprocess_data(speech_entries, noise_file_paths, dir_save)
    print('-' * 95)


def train_test(F, G, dset_loaders, criterion, epoch, phase, optimizers, args, assets, logger, use_gpu):
    if phase == 'val' or phase == 'test':
        F.eval()
        G.eval()
    if phase == 'train':
        F.train()
        G.train()
    if phase == 'train':
        logger.info('-' * 95)
        logger.info('Epoch {}/{}'.format(epoch+1, args.endepoch))
        logger.info('Current Learning rate: {}'.format(showLR(optimizers[0])))

    running_loss, running_l_nc, running_l_nn, running_l_cn, running_l_cc, running_all = \
        0., 0., 0., 0., 0., 0.
    
    print('')
    for batch_idx, (video_samples, mixed_spectrograms, ssed_spectrograms, speech_spectrograms) in \
            enumerate(dset_loaders[phase]):
        video_samples = video_samples.float()
        mixed_spectrograms = mixed_spectrograms.float()
        ssed_spectrograms = ssed_spectrograms.float()
        speech_spectrograms = speech_spectrograms.float()
        
        if use_gpu:
            if phase == 'train':
                video_samples, mixed_spectrograms, ssed_spectrograms, speech_spectrograms = \
                    Variable(video_samples.cuda()), Variable(mixed_spectrograms.cuda()), \
                    Variable(ssed_spectrograms.cuda()), Variable(speech_spectrograms.cuda())
            if phase == 'val' or phase == 'test':
                with torch.no_grad():
                    video_samples, mixed_spectrograms, ssed_spectrograms, speech_spectrograms = \
                        Variable(video_samples.cuda()), Variable(mixed_spectrograms.cuda()), \
                        Variable(ssed_spectrograms.cuda()), Variable(speech_spectrograms.cuda())
        else:
            if phase == 'train':
                video_samples, mixed_spectrograms, ssed_spectrograms, speech_spectrograms = \
                    Variable(video_samples), Variable(mixed_spectrograms), \
                    Variable(ssed_spectrograms), Variable(speech_spectrograms)
            if phase == 'val' or phase == 'test':
                with torch.no_grad():
                    video_samples, mixed_spectrograms, ssed_spectrograms, speech_spectrograms = \
                        Variable(video_samples), Variable(mixed_spectrograms), \
                        Variable(ssed_spectrograms), Variable(speech_spectrograms)

        Fx = F(video_samples, mixed_spectrograms, ssed_spectrograms)
        v_GFx, a_GFx, s_GFx = G(Fx)
        v_Gy, a_Gy, s_Gy = G(speech_spectrograms)
        FGy = F(v_Gy, a_Gy, s_Gy)

        speech_spectrograms = speech_spectrograms.contiguous()
        l1 = 0.6
        l2 = 0.4
        l3 = 1.4
        l4 = 0.8
        l_nc = criterion(Fx, speech_spectrograms)
        l_nn = criterion(a_GFx, mixed_spectrograms) + criterion(s_GFx, ssed_spectrograms) + criterion(v_GFx, video_samples)
        l_cn = criterion(a_Gy, mixed_spectrograms) + criterion(s_Gy, ssed_spectrograms) + criterion(v_Gy, video_samples)
        l_cc = criterion(FGy, speech_spectrograms)
        loss_F = l_nc + l1*l_cc
        loss_G =  l3 * (l4*loss_F + l_nn + l2*l_cn)
        #loss = l_nc + l1*l_nn + l2*l_cn + l3*l_cc

        if phase == 'train':
                optimizers[0].zero_grad()
                loss_F.backward(retain_graph=True)
                optimizers[0].step()
                optimizers[1].zero_grad()
                loss_G.backward()
                optimizers[1].step()
        # stastics
        running_l_nc += l_nc.data * video_samples.size(0)
        running_l_nn += l_nn.data * video_samples.size(0)
        running_l_cn += l_cn.data * video_samples.size(0)
        running_l_cc += l_cc.data * video_samples.size(0)
        #running_loss += loss.data * video_samples.size(0)
        running_all += len(video_samples)

        if batch_idx == 0:
            since = time.time()
        elif batch_idx % args.interval == 0 or (batch_idx == len(dset_loaders[phase])-1):
            print(
                'Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\n    L_nc: {:.4f}\tL_nn: {:.4f}\tL_cn: {:.4f}\tL_cc: {:.4f}\n    Cost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                    running_all,
                    len(dset_loaders[phase].dataset),
                    100. * batch_idx / (len(dset_loaders[phase])-1),
                    #running_loss / running_all,
                    running_l_nc / running_all,
                    running_l_nn / running_all,
                    running_l_cn / running_all,
                    running_l_cc / running_all,
                    time.time()-since,
                    (time.time()-since)*(len(dset_loaders[phase])-1) / batch_idx - (time.time()-since))
                )
    logger.info(
        '{} Epoch:\t{:2}\tL_nc: {:.4f}\tL_nn: {:.4f}\tL_cn: {:.4f}\tL_cc: {:.4f}'.format(
            phase,
            epoch+1,
            #running_loss / len(dset_loaders[phase].dataset),
            running_l_nc / len(dset_loaders[phase].dataset),
            running_l_nn / len(dset_loaders[phase].dataset),
            running_l_cn / len(dset_loaders[phase].dataset),
            running_l_cc / len(dset_loaders[phase].dataset)
            )
        )

    if phase == 'train':
        torch.save(F.state_dict(), assets.get_model_cache_path(args.model[0], epoch))
        torch.save(G.state_dict(), assets.get_model_cache_path(args.model[1], epoch))
        return F, G


def test_adam(args):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('-' * 95)
        print('GPU is available: '+torch.cuda.get_device_name(0))

    assets = AssetManager(args)
    assets.create_model(args.model[0])
    assets.create_model(args.model[1])

    # logging info
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(assets.get_logger_cache_path(args), mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    
    dset_loaders, dset_sizes, video_shape, audio_spectrogram_shape = data_loader(args, assets)

    F = network_F(video_shape, audio_spectrogram_shape)
    G = network_G(video_shape, audio_spectrogram_shape)

    print('-' * 95)
    F = reload_model(assets, args.model[0], F, args.data_model[0], logger)
    G = reload_model(assets, args.model[1], G, args.data_model[1], logger)
    
    # define loss function and optimizer
    criterion = nn.MSELoss()
    optimizers = [
        optim.Adam(
            [{'params': F.parameters(), 'lr': args.lr, 'weight_decay': 0.}],
            lr=0., weight_decay=0.),
        optim.Adam(
            [{'params': G.parameters(), 'lr': args.lr, 'weight_decay': 0.}],
            lr=0., weight_decay=0.)
    ]

    #scheduler = AdjustLR(optimizer, [args.lr, args.lr], sleep_epochs=5, half=5, verbose=1)
    if args.test == 'True':
        with torch.no_grad():
            train_test(F, G, dset_loaders, criterion, 0, 'test', optimizers, args, assets, logger,
                       use_gpu)
            print('-' * 95)
        return

    for epoch in range(args.endepoch - args.startepoch + 1):
        epoch += args.startepoch - 1
        #scheduler.step(epoch)
        F, G = train_test(F, G, dset_loaders, criterion, epoch, 'train', optimizers, args, assets,
                           logger, use_gpu)
        with torch.no_grad():
            train_test(F, G, dset_loaders, criterion, epoch, 'val', optimizers, args, assets, logger, 
                       use_gpu)
    print('-' * 95)


def reconstruct_spectrogram(args, assets, sample):
    use_gpu = torch.cuda.is_available()
    assets.create_model(args.model[0])

    video_shape, audio_spectrogram_shape = sample.video_samples.shape[1:], sample.mixed_spectrograms.shape[1:]
    model = network_F(video_shape, audio_spectrogram_shape)
    model = reload_model(assets, args.model[0], model, args.data_model[0])

    # define loss function and optimizer
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        model.eval()

        video_normalizer = make_normalizer(args, assets, args.model)
        video_normalizer.normalize(sample.video_samples)

        video_samples = torch.from_numpy(sample.video_samples)
        mixed_spectrograms = torch.from_numpy(sample.mixed_spectrograms)
        ssed_spectrograms = torch.from_numpy(sample.ssed_spectrograms)
        speech_spectrograms = torch.from_numpy(sample.speech_spectrograms)
        video_samples = video_samples.float()
        mixed_spectrograms = mixed_spectrograms.float()
        ssed_spectrograms = ssed_spectrograms.float()
        speech_spectrograms = speech_spectrograms.float()

        if use_gpu:
            with torch.no_grad():
                video_samples, mixed_spectrograms, ssed_spectrograms, speech_spectrograms = \
                    Variable(video_samples.cuda()), Variable(mixed_spectrograms.cuda()), \
                    Variable(ssed_spectrograms.cuda()), Variable(speech_spectrograms.cuda())
        else:
            with torch.no_grad():
                video_samples, mixed_spectrograms, ssed_spectrograms, speech_spectrograms = \
                    Variable(video_samples), Variable(mixed_spectrograms), \
                    Variable(ssed_spectrograms), Variable(speech_spectrograms)

        outputs = model(video_samples, mixed_spectrograms, ssed_spectrograms)
        loss = criterion(outputs, speech_spectrograms)
        
        # stastics
        running_loss, running_all = 0., 0.
        running_loss += loss.data * video_samples.size(0)
        running_all += len(video_samples)
        loss = running_loss / running_all

    return outputs, loss


def evaluate(args, assets, path_sample):
    with open(path_sample, 'rb') as fd:
	    sample = pickle.load(fd)
    outputs, loss = reconstruct_spectrogram(args, assets, sample)
    loss = loss.cpu()
    outputs = outputs.cpu()
    outputs = torch.Tensor.numpy(outputs)

    audio_signal = AudioSignal.from_wav_file(sample.speech_file_path)
    reconstruct_signal = _utils.data_processor.reconstruct_speech_signal(sample.mixed_signal, outputs, sample.video_frame_rate)

    snr = calcurate_snr(audio_signal, reconstruct_signal)
    pesq = calcurate_pesq(audio_signal, reconstruct_signal)

    return loss, snr, pesq


def try_evaluate(path_samples):
    try:
        return evaluate(*path_samples)

    except Exception as e:
        print("failed to evaluate %s (%s)" % (path_samples, e))


def make_wav(args, assets, path_sample):
    with open(path_sample, 'rb') as fd:
	    sample = pickle.load(fd)
    outputs, loss = reconstruct_spectrogram(args,assets, sample)
    outputs = outputs.cpu()
    outputs = torch.Tensor.numpy(outputs)

    audio_signal = AudioSignal.from_wav_file(sample.speech_file_path)

    reconstruct_signal = _utils.data_processor.reconstruct_speech_signal(sample.mixed_signal, outputs, sample.video_frame_rate)
    print('-' * 95)

    spath_audio = assets.get_copy_cache_path(args, sample.speech_file_path)
    print('saving original audio...')
    shutil.copyfile(sample.speech_file_path, spath_audio)
    spath_mixed = assets.get_wav_cache_path(args, sample.speech_file_path, 'mixed')
    print('saving mixed audio...')
    sample.mixed_signal.save_to_wav_file(spath_mixed)
    spath_ssed = assets.get_wav_cache_path(args, sample.speech_file_path, 'ssed')
    print('saving ssed audio...')
    sample.ssed_signal.save_to_wav_file(spath_ssed)
    spath_recon = assets.get_wav_cache_path(args, sample.speech_file_path, 'recon')
    print('saving reconstructed audio...')
    reconstruct_signal.save_to_wav_file(spath_recon)


def evaluator(args):
    #print('-' * 95)
    #print('evaluating...')
    
    assets = AssetManager(args)
    preprocessed_blob_path = assets.get_preprocessed_blob_path(args.data_evaluate[0])
    path_samples = os.path.join(preprocessed_blob_path, 'samples', '*.pkl')
    path_samples = glob.glob(path_samples)
    args = [args]*len(path_samples)
    assets = [assets]*len(path_samples)
    ziped_args = zip(args, assets, path_samples)
    thread_pool = multiprocessing.Pool(4)

    """
    with tqdm(total=len(args), ncols=95) as t:
        losses = 0.
        snrs = 0.
        pesqs = 0.
        for loss, snr, pesq in thread_pool.imap_unordered(try_evaluate, ziped_args):
            losses += loss
            snrs += snr
            pesqs += pesq
            t.update(1)

    loss = losses / len(args)
    snr = snrs / len(args)
    pesq = pesqs / len(args)

    print('loss: {:.4f}\tsnr: {:.4f}\tpesq: {:.4f}\t'.format(loss, snr, pesq))
    """
    make_wav(args[0], assets[0], path_samples[0])
    print('-' * 95)


def reconstruct_signal(args, assets, sample):
    alpha = 2
    n_fft = 640
    hop_length = 160

    mixed_signal = sample.mixed_signal
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


def ss_(args, assets, path_sample):
    with open(path_sample, 'rb') as fd:
	    sample = pickle.load(fd)
    outputs = reconstruct_signal(args, assets, sample)
    audio_signal = AudioSignal.from_wav_file(sample.speech_file_path)
    
    snr = calcurate_snr(audio_signal, outputs)
    pesq = calcurate_pesq(audio_signal, outputs)

    return snr, pesq


def try_ss(path_samples):
    try:
        return spectral_subtract(*path_samples)

    except Exception as e:
        print("failed to ss %s (%s)" % (path_samples, e))


def make_wav_ss(args, assets, path_sample):
    with open(path_sample, 'rb') as fd:
	    sample = pickle.load(fd)
    reconstructed = reconstruct_signal(args,assets, sample)
    audio_signal = AudioSignal.from_wav_file(sample.speech_file_path)

    print('-' * 95)

    spath_audio = assets.get_copy_cache_path(args, sample.speech_file_path)
    print('saving original audio...')
    shutil.copyfile(sample.speech_file_path, spath_audio)
    spath_mixed = assets.get_wav_cache_path(args, sample.speech_file_path, 'mixed')
    print('saving mixed audio...')
    sample.mixed_signal.save_to_wav_file(spath_mixed)
    spath_recon = assets.get_wav_cache_path(args, sample.speech_file_path, 'recon')
    print('saving reconstructed audio...')
    reconstructed.save_to_wav_file(spath_recon)


def ss(args):
    print('-' * 95)
    print('Doing spectral subtraction...')
    assets = AssetManager(args)
    preprocessed_blob_path = assets.get_preprocessed_blob_path(args.data_evaluate[0])
    path_samples = os.path.join(preprocessed_blob_path, 'samples', '*.pkl')
    path_samples = glob.glob(path_samples)
    args = [args]*len(path_samples)
    assets = [assets]*len(path_samples)
    ziped_args = zip(args, assets, path_samples)
    thread_pool = multiprocessing.Pool(20)

    with tqdm(total=len(args), ncols=95) as t:
        snrs = 0.
        pesqs = 0.
        for snr, pesq in thread_pool.imap_unordered(try_ss, ziped_args):
            snrs += snr
            pesqs += pesq
            t.update(1)

    snr = snrs / len(args)
    pesq = pesqs / len(args)

    print('snr: {:.4f}\tpesq: {:.4f}\t'.format(snr, pesq))
    
    make_wav_ss(args[0], assets[0], path_samples[0])

    for i in range(len(args)):
        make_wav_ss(args[i], assets[i], path_samples[i])

    print('-' * 95)


def main():
    sys.stdout.flush()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--base_dir', type=str, required=True)

    action_parsers = parser.add_subparsers()
    preprocess_parser = action_parsers.add_parser('preprocess')
    preprocess_parser.add_argument('--data_dir', type=str, required=True)
    preprocess_parser.add_argument('--data_name', type=str, required=True)
    preprocess_parser.add_argument('--dataset_dir', type=str, required=True)
    preprocess_parser.add_argument('--noise_dirs', nargs='+', type=str, required=True)
    preprocess_parser.add_argument('--speakers', nargs='+', type=str)
    preprocess_parser.add_argument('--ignored_speakers', nargs='+', type=str)
    preprocess_parser.set_defaults(func=preprocess)
    
    train_test_parser = action_parsers.add_parser('train_test')
    train_test_parser.add_argument('--data_dir', type=str, required=True)
    train_test_parser.add_argument('--model', type=str, nargs='+', required=True)
    train_test_parser.add_argument('--data_model', nargs='+', type=int, required=True)
    train_test_parser.add_argument('--data_train', nargs='+', type=str, required=True)
    train_test_parser.add_argument('--data_validation', nargs='+', type=str, required=True)
    train_test_parser.add_argument('--data_test', nargs='+', type=str, required=True)
    train_test_parser.add_argument('--batch-size', type=int, required=True)
    train_test_parser.add_argument('--lr', type=float)
    train_test_parser.add_argument('--workers', type=int, required=True)
    train_test_parser.add_argument('--startepoch', type=int)
    train_test_parser.add_argument('--endepoch', type=int)
    train_test_parser.add_argument('--interval', default=10, type=int)
    train_test_parser.add_argument('--test', default='False', type=str)
    train_test_parser.add_argument('--normalization', default='False', type=str)
    train_test_parser.set_defaults(func=test_adam)
    
    evaluate_parser = action_parsers.add_parser('evaluate')
    evaluate_parser.add_argument('--data_dir', type=str, required=True)
    evaluate_parser.add_argument('--model', type=str, nargs='+', required=True)
    evaluate_parser.add_argument('--data_model', nargs='+', type=int, required=True)
    evaluate_parser.add_argument('--data_evaluate', nargs='+', type=str, required=True)
    evaluate_parser.add_argument('--workers', type=int, required=True)
    evaluate_parser.add_argument('--interval', default=10, type=int)
    evaluate_parser.add_argument('--test', default='False', type=str)
    evaluate_parser.add_argument('--normalization', default='False', type=str)
    evaluate_parser.set_defaults(func=evaluator)

    ss_parser = action_parsers.add_parser('ss')
    ss_parser.add_argument('--data_dir', type=str, required=True)
    ss_parser.add_argument('--model', type=str, nargs='+', required=True)
    ss_parser.add_argument('--data_evaluate', nargs='+', type=str, required=True)
    ss_parser.add_argument('--workers', type=int, required=True)
    ss_parser.add_argument('--interval', default=10, type=int)
    ss_parser.add_argument('--test', default='False', type=str)
    ss_parser.add_argument('--normalization', default='False', type=str)
    ss_parser.set_defaults(func=ss)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
