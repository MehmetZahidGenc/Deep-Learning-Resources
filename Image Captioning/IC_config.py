"""
Config info - Created by Mehmet Zahid Genç

"""
import torchvision.transforms as transforms

# Hyperparameters
embedding_size = 256
hidden_size = 256
num_layers = 1
learning_rate = 3e-4
num_epochs = 100

batch_size = 32
num_workers = 8
root_folder = 'flickr8k/images'
annotation_file = 'flickr8k/captions.txt'

checkpoint_filename = 'my_checkpoint.pth.tar'
writer_filename = 'flickr'

load_model = False
save_model = False
train_CNN = False

transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

