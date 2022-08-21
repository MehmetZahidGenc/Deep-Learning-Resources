"""
Created by Mehmet Zahid Genç
"""

from IC_dataloader import get_loader
from IC_model import cnn_to_rnn
import IC_config
from IC_train import train_model

train_loader, dataset = get_loader() # load data

vocab_size = len(dataset.vocab) # calculate vocab size

model = cnn_to_rnn(embedding_size=IC_config.embedding_size, vocab_size=vocab_size, hidden_size=IC_config.hidden_size, num_layers=IC_config.num_layers) # creating our model

train_model(train_loader, dataset, model, IC_config.num_epochs) # train our model