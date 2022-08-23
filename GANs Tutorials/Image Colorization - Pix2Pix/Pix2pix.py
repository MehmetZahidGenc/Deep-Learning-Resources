import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from init_weights import weights_init
from discriminator_model import Discriminator
from generator_model import Generator
from custom_data_load import CustomData
import config
from train_GAN import train
from utils_GAN import save_checkpoint, save_some_examples


# Parameters
lr = config.LEARNING_RATE
batch_size = config.BATCH_SIZE
image_size = config.IMAGE_SIZE
number_of_channel = config.CHANNELS_IMG
num_epochs = config.NUM_EPOCHS

# Discriminator Network and optimizer
netD = Discriminator(in_channel=3, features=[64, 128, 256, 512])
weights_init(netD)
opt_D = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

# Generator Network and optimizer
netG = Generator(in_channel=3, feature=64)
weights_init(netG)
opt_G = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss
BCE = nn.BCEWithLogitsLoss()
L1_LOSS = nn.L1Loss()

# Scalers
g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

# Transformer
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(number_of_channel)], [0.5 for _ in range(number_of_channel)]
    ),
])

# Datasets
train_dataset = CustomData(target_root_dir='data//train_color', input_root_dir='data//train_black', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CustomData(target_root_dir='data//test_color', input_root_dir='data//test_black', transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

for epoch in range(num_epochs):
    train(netD, netG, train_dataloader, opt_D, opt_G, l1_loss=L1_LOSS, bce=BCE, g_scaler=g_scaler, d_scaler=d_scaler)

    if config.SAVE_MODEL and epoch % 5 == 0:
        save_checkpoint(netG, opt_G, filename=config.CHECKPOINT_GEN)
        save_checkpoint(netD, opt_D, filename=config.CHECKPOINT_DISC)

    save_some_examples(netG, val_dataloader, epoch, folder="evaluation")