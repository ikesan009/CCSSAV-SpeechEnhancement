"""
video_shape: [128, 128, 4]
audio_spectrogram_shape: [80, 19]

Audio_encoder:
    torch.Size([80, 19])
    torch.Size([1, 80, 19])
    torch.Size([64, 40, 10])
    torch.Size([64, 40, 10])
    torch.Size([128, 20, 5])
    torch.Size([128, 10, 5])
    torch.Size([128, 5, 5])

Video_encoder:
    torch.Size([128, 128, 4])
    torch.Size([4, 128, 128])
    torch.Size([4, 128, 128])
    torch.Size([128, 64, 64])
    torch.Size([128, 32, 32])
    torch.Size([128, 32, 32])
    torch.Size([128, 16, 16])
    torch.Size([256, 16, 16])
    torch.Size([256, 8, 8])
    torch.Size([256, 8, 8])
    torch.Size([256, 4, 4])
    torch.Size([512, 4, 4])
    torch.Size([512, 2, 2])
    torch.Size([512, 2, 2])
    torch.Size([512, 1, 1])
"""


# coding: utf-8
import torch
import torch.nn as nn
from _utils.network_encoder import *
from _utils.network_decoder import *


class F(nn.Module):
    def __init__(self, video_shape, audio_spectrogram_shape):
        super(F, self).__init__()
        self.video_shape = video_shape
        self.audio_spectrogram_shape = audio_spectrogram_shape
        self.shape_audio_embedding_matrix = [-1, 128, 5, 5]
        self.size_audio_embedding = 3200
        self.size_shared_embedding = int(3200 * 2 + 512)
        self.size_embedding = int(self.size_shared_embedding / 4)

        self.encoder = Encoder_F(self.video_shape, self.audio_spectrogram_shape,
            self.shape_audio_embedding_matrix, self.size_audio_embedding, 
            self.size_shared_embedding, self.size_embedding)
        self.decoder = Decoder_F(self.shape_audio_embedding_matrix, self.size_audio_embedding, 
        self.size_shared_embedding, self.size_embedding)

    def forward(self, video_input, audio_input, ss_input):
        x = video_input
        y = audio_input
        y2 = ss_input
        x = self.encoder(x, y, y2)
        x = self.decoder(x)
        reshaped = [-1]
        reshaped.extend(self.audio_spectrogram_shape)
        out = x.reshape(reshaped)
        return out


class G(nn.Module):
    def __init__(self, video_shape, audio_spectrogram_shape):
        super(G, self).__init__()
        self.video_shape = video_shape
        self.audio_spectrogram_shape = audio_spectrogram_shape
        self.shape_audio_embedding_matrix = [-1, 128, 5, 5]
        self.shape_video_embedding_matrix = [-1, 512, 1, 1]
        self.size_audio_embedding = 3200
        self.size_shared_embedding = int(3200 * 2 + 512)
        self.size_embedding = int(self.size_shared_embedding / 4)

        self.encoder = Encoder_G(self.audio_spectrogram_shape, self.shape_audio_embedding_matrix,
            self.size_audio_embedding, self.size_shared_embedding, self.size_embedding)
        self.decoder = Decoder_G(self.video_shape, self.shape_audio_embedding_matrix,
        self.shape_video_embedding_matrix, self.size_audio_embedding, self.size_shared_embedding,
        self.size_embedding)

    def forward(self, audio_input):
        x = audio_input
        x = self.encoder(x)
        x, y, y2 = self.decoder(x)
        reshaped = [-1]
        reshaped.extend(self.audio_spectrogram_shape)
        y = y.reshape(reshaped)
        y2 = y2.reshape(reshaped)
        return x, y, y2


def network_F(video_shape, audio_spectrogram_shape):
    model = F(video_shape, audio_spectrogram_shape)
    return model


def network_G(video_shape, audio_spectrogram_shape):
    model = G(video_shape, audio_spectrogram_shape)
    return model