import torch
import torch.nn as nn
import torch.nn.functional as F

# Generator Class

def generator_conv(in_channel, out_channel, kernel_size=3, stride=2,padding=1, batch_norm=True, bias=False):
    layers = [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channel))

    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, num_of_z, number_of_channel, generator_dim):
        super(Generator, self).__init__()

        self.gen_conv1 = generator_conv(num_of_z, generator_dim * 8, kernel_size=4, stride=1, padding=0, batch_norm=True)
        self.gen_conv2 = generator_conv(generator_dim * 8, generator_dim * 4, kernel_size=4, stride=2 , padding=1, batch_norm=True)
        self.gen_conv3 = generator_conv(generator_dim * 4, generator_dim * 2, kernel_size=4, stride=2, padding=1, batch_norm=True)
        self.gen_conv4 = generator_conv(generator_dim * 2, generator_dim, kernel_size=4, stride=2, padding=1, batch_norm=True)

        self.gen_conv5 = generator_conv(generator_dim, number_of_channel, kernel_size=4, stride=2, padding=1, batch_norm=False)


    def forward(self, mzg):
        out = F.relu(self.gen_conv1(mzg))
        out = F.relu(self.gen_conv2(out))
        out = F.relu(self.gen_conv3(out))
        out = F.relu((self.gen_conv4(out)))

        out = torch.tanh(self.gen_conv5(out))

        return out