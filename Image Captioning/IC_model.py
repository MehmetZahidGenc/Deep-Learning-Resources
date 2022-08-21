"""
Created by Mehmet Zahid Genç
"""

import torch
import torch.nn as nn
import torchvision.models as models


class encoderCNN(nn.Module):
    def __init__(self, embedding_size, cnn_train=False, dropout=0.50):
        super(encoderCNN, self).__init__()

        self.cnn_train = cnn_train
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, mzg):
        features = self.inception(mzg)

        for name, param in self.inception.named_parameters():
            param.requires_grad = False

        return self.dropout(self.relu(features))


class decoderRNN(nn.Module):
    def __init__(self, embedding_size, vocab_size, hidden_size, num_layers, dropout=0.50):
        super(decoderRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, caption):
        embeddings = self.dropout(self.embedding(caption))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)

        return outputs

class cnn_to_rnn(nn.Module):
    def __init__(self, embedding_size, vocab_size, hidden_size, num_layers):
        super(cnn_to_rnn, self).__init__()
        self.encoderCNN = encoderCNN(embedding_size)
        self.decoderRNN = decoderRNN(embedding_size, vocab_size, hidden_size, num_layers)

    def forward(self, mzg, caption):
        output = self.encoderCNN(mzg)
        output = self.decoderRNN(output, caption)

        return output

    def image_caption(self, image, vocab, max_length=50):
        caption_results = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                print(predicted.shape)
                caption_results.append(predicted.item())
                x = self.decoderRNN.embedding(output).unsqueeze(0)

                if vocab.itos[predicted.item()] == "<EOS>":
                    break
            return [vocab.itos[i] for i in caption_results]