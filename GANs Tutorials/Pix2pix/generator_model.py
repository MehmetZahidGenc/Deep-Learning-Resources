# class Generator

import torch.nn as nn
import torch
import torch.nn.functional as F


def generator_conv(in_channel, out_channel, kernel_size=3, stride=2, padding=1, batch_norm=True, bias=False,
                   type='encoder'):
    layers = []

    if type == 'encoder':
        layers = [
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channel))
        else:
            pass


    elif type == 'decoder':
        layers = [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                                     bias=bias)]

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channel))
        else:
            pass

    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, in_channel=3, feature=64):
        super(Generator, self).__init__()

        # U-NET encoder part
        self.en_conv1 = generator_conv(in_channel, feature, kernel_size=4, stride=2, padding=1, batch_norm=False,
                                       bias=False, type='encoder')
        self.en_conv2 = generator_conv(feature, feature * 2, kernel_size=4, stride=2, padding=1, batch_norm=True,
                                       bias=False, type='encoder')
        self.en_conv3 = generator_conv(feature * 2, feature * 4, kernel_size=4, stride=2, padding=1, batch_norm=True,
                                       bias=False, type='encoder')
        self.en_conv4 = generator_conv(feature * 4, feature * 8, kernel_size=4, stride=2, padding=1, batch_norm=True,
                                       bias=False, type='encoder')
        self.en_conv5 = generator_conv(feature * 8, feature * 8, kernel_size=4, stride=2, padding=1, batch_norm=True,
                                       bias=False, type='encoder')
        self.en_conv6 = generator_conv(feature * 8, feature * 8, kernel_size=4, stride=2, padding=1, batch_norm=True,
                                       bias=False, type='encoder')
        self.en_conv7 = generator_conv(feature * 8, feature * 8, kernel_size=4, stride=2, padding=1, batch_norm=True,
                                       bias=False, type='encoder')

        self.en_conv8 = generator_conv(feature * 8, feature * 8, kernel_size=4, stride=2, padding=1, batch_norm=False,
                                       bias=False, type='encoder')

        # U-Net decoder part
        self.dec_conv1 = generator_conv(feature * 8, feature * 8, kernel_size=4, stride=2, padding=1, batch_norm=True,
                                        bias=False, type='decoder')
        self.dec_conv2 = generator_conv(feature * 8 * 2, feature * 8, kernel_size=4, stride=2, padding=1,
                                        batch_norm=True, bias=False, type='decoder')
        self.dec_conv3 = generator_conv(feature * 8 * 2, feature * 8, kernel_size=4, stride=2, padding=1,
                                        batch_norm=True, bias=False, type='decoder')
        self.dec_conv4 = generator_conv(feature * 8 * 2, feature * 8, kernel_size=4, stride=2, padding=1,
                                        batch_norm=True, bias=False, type='decoder')
        self.dec_conv5 = generator_conv(feature * 8 * 2, feature * 4, kernel_size=4, stride=2, padding=1,
                                        batch_norm=True, bias=False, type='decoder')
        self.dec_conv6 = generator_conv(feature * 4 * 2, feature * 2, kernel_size=4, stride=2, padding=1,
                                        batch_norm=True, bias=False, type='decoder')
        self.dec_conv7 = generator_conv(feature * 2 * 2, feature, kernel_size=4, stride=2, padding=1, batch_norm=True,
                                        bias=False, type='decoder')

        self.dec_conv8 = generator_conv(feature * 2, in_channel, kernel_size=4, stride=2, padding=1, batch_norm=False,
                                        bias=False, type='decoder')

    def forward(self, input):
        # encoder forward
        out_enc1 = F.leaky_relu(self.en_conv1(input), 0.2)
        out_enc2 = F.leaky_relu(self.en_conv2(out_enc1), 0.2)
        out_enc3 = F.leaky_relu(self.en_conv3(out_enc2), 0.2)
        out_enc4 = F.leaky_relu(self.en_conv4(out_enc3), 0.2)
        out_enc5 = F.leaky_relu(self.en_conv5(out_enc4), 0.2)
        out_enc6 = F.leaky_relu(self.en_conv6(out_enc5), 0.2)
        out_enc7 = F.leaky_relu(self.en_conv7(out_enc6), 0.2)
        out_enc8 = F.relu(self.en_conv8(out_enc7))

        # decoder forward

        out_dec1 = F.dropout(self.dec_conv1(out_enc8), 0.5, training=True)
        out_dec1 = torch.cat([out_dec1, out_enc7], dim=1)
        out_dec1 = F.relu(out_dec1)

        out_dec2 = F.dropout(self.dec_conv2(out_dec1), 0.5, training=True)
        out_dec2 = torch.cat([out_dec2, out_enc6], dim=1)
        out_dec2 = F.relu(out_dec2)

        out_dec3 = F.dropout(self.dec_conv3(out_dec2), 0.5, training=True)
        out_dec3 = torch.cat([out_dec3, out_enc5], dim=1)
        out_dec3 = F.relu(out_dec3)

        out_dec4 = self.dec_conv4(out_dec3)
        out_dec4 = torch.cat([out_dec4, out_enc4], dim=1)
        out_dec4 = F.relu(out_dec4)

        out_dec5 = self.dec_conv5(out_dec4)
        out_dec5 = torch.cat([out_dec5, out_enc3], dim=1)
        out_dec5 = F.relu(out_dec5)

        out_dec6 = self.dec_conv6(out_dec5)
        out_dec6 = torch.cat([out_dec6, out_enc2], dim=1)
        out_dec6 = F.relu(out_dec6)

        out_dec7 = self.dec_conv7(out_dec6)
        out_dec7 = torch.cat([out_dec7, out_enc1], dim=1)
        out_dec7 = F.relu(out_dec7)

        out_dec8 = self.dec_conv8(out_dec7)

        output = torch.tanh(out_dec8)

        return output