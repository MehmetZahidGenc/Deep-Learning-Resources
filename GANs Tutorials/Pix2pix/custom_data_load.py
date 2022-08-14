import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

class CustomData(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, item):
        img_file = self.list_files[item]
        img_path = os.path.join(self.root_dir, img_file)

        image = np.array(Image.open(img_path))

        input_image = image[:, :256, :]
        input_image = Image.fromarray(input_image)

        target_image = image[:, 256:, :]
        target_image = Image.fromarray(target_image)

        input_image = self.transform(input_image)
        target_image = self.transform(target_image)

        return input_image, target_image