from PIL import Image
import os
from torch.utils.data import Dataset


class CustomData(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.images = os.listdir(data_dir)

        self.images_len = len(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = self.images[item % self.images_len]

        img_path = os.path.join(self.data_dir, img)

        img = Image.open(img_path).convert("RGB")

        img = self.transform(img)

        return img

