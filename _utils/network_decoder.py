# coding: utf-8
import torch
import torch.nn as nn
from _utils.util_network import *


class Audio_Decoder(nn.Module):
    def __init__(self):
        super(Audio_Decoder, self).__init__()
        self.deconv_1 = self.deconv2d(128, 128, size_kernel=(2, 3), stride=(2, 1), BN=True, Relu=True)
        self.deconv_2 = self.deconv2d(128, 128, size_kernel=(2, 3), stride=(2, 1), BN=True, Relu=True)
        self.deconv_3 = self.deconv2d(128, 128, size_kernel=(4, 4), stride=(2, 2), BN=True, Relu=True)
        self.deconv_4 = self.deconv2d(128, 64, size_kernel=(5, 5), stride=(1, 1), BN=True, Relu=True)
        self.deconv_5 = self.deconv2d(64, 64, size_kernel=(6, 5), stride=(2, 2), BN=True, Relu=True)
        self.deconv_6 = self.deconv2d(64, 1, size_kernel=(1, 1), stride=(1, 1), BN=False, Relu=False)

    def forward(self, x):
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.deconv_4(x)
        x = self.deconv_5(x)
        x = self.deconv_6(x)
        return x

    def deconv2d(self, channel_i, channel_o, size_kernel, stride, BN, Relu):
        padding = []
        for i in range(2):
            padding.append(same_padding(size_kernel[i], stride[i]))
        padding = tuple(padding)
        layers = []
        layers.append(nn.ConvTranspose2d(channel_i, channel_o, kernel_size=size_kernel, stride=stride,
            padding=padding))
        if BN:
            layers.append(nn.BatchNorm2d(channel_o))
        if Relu:
            layers.append(nn.LeakyReLU(0.3, True))
        return nn.Sequential(*layers)


class Video_Decoder(nn.Module):
    def __init__(self, video_shape):
        super(Video_Decoder, self).__init__()
        self.deconv_1_1 = self.deconv2d(512, 512, size_kernel=(2, 2), stride=(2, 2), BN=True, Relu=True)
        self.deconv_1_2 = self.deconv2d(512, 512, size_kernel=(3, 3), stride=(1, 1), BN=True, Relu=True)
        self.deconv_2_1 = self.deconv2d(512, 512, size_kernel=(2, 2), stride=(2, 2), BN=True, Relu=True)
        self.deconv_2_2 = self.deconv2d(512, 256, size_kernel=(3, 3), stride=(1, 1), BN=True, Relu=True)
        self.deconv_3_1 = self.deconv2d(256, 256, size_kernel=(2, 2), stride=(2, 2), BN=True, Relu=True)
        self.deconv_3_2 = self.deconv2d(256, 256, size_kernel=(3, 3), stride=(1, 1), BN=True, Relu=True)
        self.deconv_4_1 = self.deconv2d(256, 256, size_kernel=(2, 2), stride=(2, 2), BN=True, Relu=True)
        self.deconv_4_2 = self.deconv2d(256, 128, size_kernel=(3, 3), stride=(1, 1), BN=True, Relu=True)
        self.deconv_5_1 = self.deconv2d(128, 128, size_kernel=(2, 2), stride=(2, 2), BN=True, Relu=True)
        self.deconv_5_2 = self.deconv2d(128, 128, size_kernel=(3, 3), stride=(1, 1), BN=True, Relu=True)
        self.deconv_6_1 = self.deconv2d(128, 128, size_kernel=(2, 2), stride=(2, 2), BN=True, Relu=True)
        self.deconv_6_2 = self.deconv2d(128, video_shape[2], size_kernel=(4, 4), stride=(2, 2), BN=False, Relu=False)

    def forward(self, x):
        x = self.deconv_1_1(x)
        x = self.deconv_1_2(x)
        x = self.deconv_2_1(x)
        x = self.deconv_2_2(x)
        x = self.deconv_3_1(x)
        x = self.deconv_3_2(x)
        x = self.deconv_4_1(x)
        x = self.deconv_4_2(x)
        x = self.deconv_5_1(x)
        x = self.deconv_5_2(x)
        x = self.deconv_6_1(x)
        x = self.deconv_6_2(x)
        x = x.transpose(1, 3)
        x = x.transpose(1, 2)
        return x

    def deconv2d(self, channel_i, channel_o, size_kernel, stride, BN, Relu):
        padding = []
        for i in range(2):
            padding.append(same_padding(size_kernel[i], stride[i]))
        padding = tuple(padding)
        layers = []
        layers.append(nn.ConvTranspose2d(channel_i, channel_o, kernel_size=size_kernel, stride=stride,
            padding=padding))
        if BN:
            layers.append(nn.BatchNorm2d(channel_o))
        if Relu:
            layers.append(nn.LeakyReLU(0.3, True))
        return nn.Sequential(*layers)


class Decoder_F(nn.Module):
    def __init__(self, shape_audio_embedding_matrix, size_audio_embedding, 
        size_shared_embedding, size_embedding):
        super(Decoder_F, self).__init__()
        self.shape_audio_embedding_matrix = shape_audio_embedding_matrix
        self.size_audio_embedding = size_audio_embedding
        self.size_shared_embedding = size_shared_embedding
        self.size_embedding = size_embedding

        self.fc_1 = self.linear(self.size_embedding, self.size_shared_embedding, BN=True, Relu=True)
        self.fc_2 = self.linear(self.size_shared_embedding, self.size_audio_embedding, BN=False,
            Relu=False)
        self.bn = nn.BatchNorm2d(self.shape_audio_embedding_matrix[1])
        self.Relu = nn.LeakyReLU(0.3, True)
        self.audio_decoder = Audio_Decoder()

    def forward(self, shared_embedding):
        r_shared_embedding = self.fc_1(shared_embedding)
        r_audio_embedding = self.fc_2(r_shared_embedding)
        r_audio_embedding_matrix = r_audio_embedding.reshape(self.shape_audio_embedding_matrix)
        r_audio_embedding_matrix = self.bn(r_audio_embedding_matrix)
        r_audio_embedding_matrix = self.Relu(r_audio_embedding_matrix)
        audio_output = self.audio_decoder(r_audio_embedding_matrix)
        return audio_output
    
    def linear(self, size_i, size_o, BN, Relu):
        layers = []
        layers.append(nn.Linear(size_i, size_o))
        if BN:
            layers.append(nn.BatchNorm1d(size_o))
        if Relu:
            layers.append(nn.LeakyReLU(0.3, True))
        return nn.Sequential(*layers)


class Decoder_G(nn.Module):
    def __init__(self, video_shape, shape_audio_embedding_matrix, shape_video_embedding_matrix,
        size_audio_embedding, size_shared_embedding, size_embedding):
        super(Decoder_G, self).__init__()
        self.shape_audio_embedding_matrix = shape_audio_embedding_matrix
        self.shape_video_embedding_matrix = shape_video_embedding_matrix
        self.size_audio_embedding = size_audio_embedding
        self.size_shared_embedding = size_shared_embedding
        self.size_embedding = size_embedding

        self.fc_1 = self.linear(self.size_embedding, self.size_shared_embedding, BN=True, Relu=True)
        self.bn_audio = nn.BatchNorm2d(self.shape_audio_embedding_matrix[1])
        self.bn_video = nn.BatchNorm2d(self.shape_video_embedding_matrix[1])
        self.Relu = nn.LeakyReLU(0.3, True)
        self.audio_decoder = Audio_Decoder()
        self.ss_decoder = Audio_Decoder()
        self.video_decoder = Video_Decoder(video_shape)

    def forward(self, shared_embedding):
        r_shared_embedding = self.fc_1(shared_embedding)
        r_audio_embedding, r_video_embedding = \
            torch.split(r_shared_embedding, int(self.size_audio_embedding*2), dim=1)
        r_audio_embedding, r_ss_embedding = \
            torch.split(r_audio_embedding, self.size_audio_embedding, dim=1)
        r_audio_embedding_matrix = r_audio_embedding.reshape(self.shape_audio_embedding_matrix)
        r_ss_embedding_matrix = r_ss_embedding.reshape(self.shape_audio_embedding_matrix)
        r_video_embedding_matrix = r_video_embedding.reshape(self.shape_video_embedding_matrix)
        r_audio_embedding_matrix = self.bn_audio(r_audio_embedding_matrix)
        r_audio_embedding_matrix = self.Relu(r_audio_embedding_matrix)
        r_ss_embedding_matrix = self.bn_audio(r_ss_embedding_matrix)
        r_ss_embedding_matrix = self.Relu(r_ss_embedding_matrix)
        r_video_embedding_matrix = self.bn_video(r_video_embedding_matrix)
        r_video_embedding_matrix = self.Relu(r_video_embedding_matrix)
        audio_output = self.audio_decoder(r_audio_embedding_matrix)
        ss_output = self.ss_decoder(r_audio_embedding_matrix)
        video_output = self.video_decoder(r_video_embedding_matrix)
        return video_output, audio_output, ss_output

    def linear(self, size_i, size_o, BN, Relu):
        layers = []
        layers.append(nn.Linear(size_i, size_o))
        if BN:
            layers.append(nn.BatchNorm1d(size_o))
        if Relu:
            layers.append(nn.LeakyReLU(0.3, True))
        return nn.Sequential(*layers)