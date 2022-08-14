# class Discriminator

import torch.nn as nn
import torch
import torch.nn.functional as F


def disc_conv(in_channel, out_channel, kernel_size=3, stride=2, padding=1, padding_mode="reflect", batch_norm=True,
              bias=False, first_layer=False):
    if first_layer:
        layers = [
            nn.Conv2d(in_channel * 2, out_channel, kernel_size, stride=stride, padding=padding, padding_mode="reflect",
                      bias=bias)]
    else:
        layers = [
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, padding_mode="reflect",
                      bias=bias)]

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channel))

    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self, in_channel=3, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()

        self.disc_conv1 = disc_conv(in_channel, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect',
                                    batch_norm=False, bias=False, first_layer=True)
        self.disc_conv2 = disc_conv(features[0], features[1], kernel_size=4, stride=2, padding=1,
                                    padding_mode='reflect', batch_norm=True, bias=False, first_layer=False)
        self.disc_conv3 = disc_conv(features[1], features[2], kernel_size=4, stride=2, padding=1,
                                    padding_mode='reflect', batch_norm=True, bias=False, first_layer=False)
        self.disc_conv4 = disc_conv(features[2], features[3], kernel_size=4, stride=2, padding=1,
                                    padding_mode='reflect', batch_norm=True, bias=False, first_layer=False)

        self.disc_conv5 = disc_conv(features[3], 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect',
                                    batch_norm=False, bias=False, first_layer=False)

    def forward(self, input, label):
        out = torch.cat([input, label], dim=1)

        out = F.leaky_relu(self.disc_conv1(out), 0.2)
        out = F.leaky_relu(self.disc_conv2(out), 0.2)
        out = F.leaky_relu(self.disc_conv3(out), 0.2)
        out = F.leaky_relu(self.disc_conv4(out), 0.2)

        out = torch.sigmoid(self.disc_conv5(out))

        return out