import torch
import torch.nn as nn
import torch.nn.functional as F


# Discriminator Class

def discriminator_conv(in_channel, out_channel, kernel_size=3, stride=2, padding=1, batch_norm=True, bias=False):
    layers = [nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias)]

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channel))

    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self, number_of_channel, discriminator_dim):
        super(Discriminator, self).__init__()

        self.disc_conv1 = discriminator_conv(number_of_channel, discriminator_dim, kernel_size=4, stride=2, padding=1, batch_norm=False)
        self.disc_conv2 = discriminator_conv(discriminator_dim, discriminator_dim * 2, kernel_size=4, stride=2, padding=1, batch_norm=True)
        self.disc_conv3 = discriminator_conv(discriminator_dim * 2, discriminator_dim * 4, kernel_size=4, stride=2, padding=1, batch_norm=True)
        self.disc_conv4 = discriminator_conv(discriminator_dim * 4, discriminator_dim * 8, kernel_size=4, stride=2, padding=1, batch_norm=True)

        self.disc_conv5 = discriminator_conv(discriminator_dim * 8, 1, kernel_size=4, stride=1, padding=0, batch_norm=False)


    def forward(self, mzg):
        out = F.leaky_relu(self.disc_conv1(mzg), inplace=True, negative_slope=0.2)
        out = F.leaky_relu(self.disc_conv2(out), inplace=True, negative_slope=0.2)
        out = F.leaky_relu(self.disc_conv3(out), inplace=True, negative_slope=0.2)
        out = F.leaky_relu(self.disc_conv4(out), inplace=True, negative_slope=0.2)

        out = torch.sigmoid(self.disc_conv5(out))

        return out