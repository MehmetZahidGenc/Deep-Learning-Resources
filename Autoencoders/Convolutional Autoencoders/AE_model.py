import torch.nn as nn

def reversList(layers_list):
    layers_list.reverse()
    return layers_list


class AE(nn.Module):
    def __init__(self, in_channel, features=[16, 32, 64]):
        super(AE, self).__init__()
        self.features = features
        self.in_channel = in_channel

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        # encoder
        self.layers_encoder = []
        for i in range(len(self.features)):
            if i == 0:
                self.layers_encoder.append(nn.Conv2d(in_channel, features[i], kernel_size=3, stride=2, padding=1))
                self.layers_encoder.append(nn.ReLU())

            else:
                if i != len(self.features)-1:
                    self.layers_encoder.append(nn.Conv2d(features[i-1], features[i], kernel_size=3, stride=2, padding=1))
                    self.layers_encoder.append(nn.ReLU())
                elif i == len(self.features)-1:
                    self.layers_encoder.append(nn.Conv2d(features[i-1], features[i], kernel_size=7))

        # decoder
        self.layers_decoder = []
        self.features = reversList(self.features)
        for j in range(len(self.features)):
            if j == len(self.features)-1:
                self.layers_decoder.append(nn.ConvTranspose2d(self.features[j], in_channel, kernel_size=3, stride=2, padding=1, output_padding=1))
                self.layers_decoder.append((nn.Sigmoid()))
            else:
                if j == 0:
                    self.layers_decoder.append(nn.ConvTranspose2d(self.features[j], self.features[j+1], kernel_size=7))
                    self.layers_decoder.append(nn.ReLU())
                else:
                    self.layers_decoder.append(nn.ConvTranspose2d(self.features[j], self.features[j+1], kernel_size=3, stride=2, padding=1, output_padding=1))
                    self.layers_decoder.append(nn.ReLU())

        self.encoder_()
        self.decoder_()

    def encoder_(self):
        self.encoder = nn.Sequential(*self.layers_encoder)

    def decoder_(self):
        self.decoder = nn.Sequential(*self.layers_decoder)

    def forward(self, mzg):
        encoded = self.encoder(mzg)
        decoded = self.decoder(encoded)
        return decoded