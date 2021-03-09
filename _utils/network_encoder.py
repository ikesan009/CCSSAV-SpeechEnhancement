# coding: utf-8
import torch
import torch.nn as nn
from _utils.util_network import *


class Audio_Encoder(nn.Module):
    def __init__(self, audio_spectrogram_shape):
        super(Audio_Encoder, self).__init__()
        self.reshaped = [-1]
        self.reshaped.append(1)
        self.reshaped.extend(audio_spectrogram_shape)
        self.conv_1 = self.conv2d(1, 64, size_kernel=(5, 5), stride=(2, 2), BN=True, Relu=True)
        self.conv_2 = self.conv2d(64, 64, size_kernel=(5, 5), stride=(1, 1), BN=True, Relu=True)
        self.conv_3 = self.conv2d(64, 128, size_kernel=(4, 4), stride=(2, 2), BN=True, Relu=True)
        self.conv_4 = self.conv2d(128, 128, size_kernel=(2, 3), stride=(2, 1), BN=True, Relu=True)
        self.conv_5 = self.conv2d(128, 128, size_kernel=(2, 3), stride=(2, 1), BN=True, Relu=True)

    def forward(self, x):
        x = x.reshape(self.reshaped)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        return x

    def conv2d(self, channel_i, channel_o, size_kernel, stride, BN, Relu):
        padding = []
        for i in range(2):
            padding.append(same_padding(size_kernel[i], stride[i]))
        padding = tuple(padding)
        layers = []
        layers.append(nn.Conv2d(channel_i, channel_o, kernel_size=size_kernel, stride=stride, padding=padding))
        if BN:
            layers.append(nn.BatchNorm2d(channel_o))
        if Relu:
            layers.append(nn.LeakyReLU(0.3, True))
        return nn.Sequential(*layers)


class Video_Encoder(nn.Module):
    def __init__(self, video_shape):
        super(Video_Encoder, self).__init__()
        self.conv_1 = self.conv2d(video_shape[2], 128, size_kernel=(5, 5), stride=(2, 2), BN=True,
            Relu=True)
        self.pool_1 = self.maxpl2d(size_kernel=(2, 2), stride=(2, 2), Dropout=True)
        self.conv_2 = self.conv2d(128, 128, size_kernel=(5, 5), stride=(1, 1), BN=True, Relu=True)
        self.pool_2 = self.maxpl2d(size_kernel=(2, 2), stride=(2, 2), Dropout=True)
        self.conv_3 = self.conv2d(128, 256, size_kernel=(3, 3), stride=(1, 1), BN=True, Relu=True)
        self.pool_3 = self.maxpl2d(size_kernel=(2, 2), stride=(2, 2), Dropout=True)
        self.conv_4 = self.conv2d(256, 256, size_kernel=(3, 3), stride=(1, 1), BN=True, Relu=True)
        self.pool_4 = self.maxpl2d(size_kernel=(2, 2), stride=(2, 2), Dropout=True)
        self.conv_5 = self.conv2d(256, 512, size_kernel=(3, 3), stride=(1, 1), BN=True, Relu=True)
        self.pool_5 = self.maxpl2d(size_kernel=(2, 2), stride=(2, 2), Dropout=True)
        self.conv_6 = self.conv2d(512, 512, size_kernel=(3, 3), stride=(1, 1), BN=True, Relu=True)
        self.pool_6 = self.maxpl2d(size_kernel=(2, 2), stride=(2, 2), Dropout=True)

    def forward(self, x):
        x = x.transpose(1, 3)
        x = x.transpose(2, 3)
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.conv_3(x)
        x = self.pool_3(x)
        x = self.conv_4(x)
        x = self.pool_4(x)
        x = self.conv_5(x)
        x = self.pool_5(x)
        x = self.conv_6(x)
        x = self.pool_6(x)
        return x

    def conv2d(self, channel_i, channel_o, size_kernel, stride, BN, Relu):
        padding = []
        for i in range(2):
            padding.append(same_padding(size_kernel[i], stride[i]))
        padding = tuple(padding)
        layers = []
        layers.append(nn.Conv2d(channel_i, channel_o, kernel_size=size_kernel, stride=stride, padding=padding))
        if BN:
            layers.append(nn.BatchNorm2d(channel_o))
        if Relu:
            layers.append(nn.LeakyReLU(0.3, True))
        return nn.Sequential(*layers)

    def maxpl2d(self, size_kernel, stride, Dropout):
        padding = []
        for i in range(2):
            padding.append(same_padding(size_kernel[i], stride[i]))
        padding = tuple(padding)
        layers = []
        layers.append(nn.MaxPool2d(kernel_size=size_kernel, stride=stride, padding=padding))
        if Dropout:
            layers.append(nn.Dropout(p=0.25, inplace=True))
        return nn.Sequential(*layers)


class Encoder_F(nn.Module):
    def __init__(self, video_shape, audio_spectrogram_shape, shape_audio_embedding_matrix,
        size_audio_embedding, size_shared_embedding, size_embedding):
        super(Encoder_F, self).__init__()
        self.shape_audio_embedding_matrix = shape_audio_embedding_matrix
        self.size_audio_embedding = size_audio_embedding
        self.size_shared_embedding = size_shared_embedding
        self.size_embedding = size_embedding
        self.audio_encoder = Audio_Encoder(audio_spectrogram_shape)
        self.ss_encoder = Audio_Encoder(audio_spectrogram_shape)
        self.video_encoder = Video_Encoder(video_shape)
        self.fc = self.linear(self.size_shared_embedding, self.size_embedding, BN=True, Relu=True)

    def forward(self, x, y, y2):
        audio_embedding_matrix = self.audio_encoder(y)
        ss_embedding_matrix = self.ss_encoder(y2)
        video_embedding_matrix = self.video_encoder(x)
        audio_embedding = audio_embedding_matrix.reshape(audio_embedding_matrix.shape[0], -1)
        ss_embedding = ss_embedding_matrix.reshape(ss_embedding_matrix.shape[0], -1)
        video_embedding = video_embedding_matrix.reshape(video_embedding_matrix.shape[0], -1)
        shared_embedding = torch.cat([audio_embedding, ss_embedding, video_embedding], dim=1)
        shared_embedding = self.fc(shared_embedding)
        return shared_embedding

    def linear(self, size_i, size_o, BN, Relu):
        layers = []
        layers.append(nn.Linear(size_i, size_o))
        if BN:
            layers.append(nn.BatchNorm1d(size_o))
        if Relu:
            layers.append(nn.LeakyReLU(0.3, True))
        return nn.Sequential(*layers)


class Encoder_G(nn.Module):
    def __init__(self, audio_spectrogram_shape, shape_audio_embedding_matrix,
        size_audio_embedding, size_shared_embedding, size_embedding):
        super(Encoder_G, self).__init__()
        self.shape_audio_embedding_matrix = shape_audio_embedding_matrix
        self.size_audio_embedding = size_audio_embedding
        self.size_shared_embedding = size_shared_embedding
        self.size_embedding = size_embedding
        self.audio_encoder = Audio_Encoder(audio_spectrogram_shape)
        self.fc = self.linear(self.size_audio_embedding, self.size_embedding, BN=True, Relu=True)

    def forward(self, x):
        audio_embedding_matrix = self.audio_encoder(x)
        audio_embedding = audio_embedding_matrix.reshape(audio_embedding_matrix.shape[0], -1)
        shared_embedding = self.fc(audio_embedding)
        return shared_embedding

    def linear(self, size_i, size_o, BN, Relu):
        layers = []
        layers.append(nn.Linear(size_i, size_o))
        if BN:
            layers.append(nn.BatchNorm1d(size_o))
        if Relu:
            layers.append(nn.LeakyReLU(0.3, True))
        return nn.Sequential(*layers)
