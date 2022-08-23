import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

class CustomData(Dataset):
    def __init__(self, target_root_dir, input_root_dir, transform):
        self.target_root_dir = target_root_dir
        self.input_root_dir = input_root_dir

        self.transform = transform

        self.list_files_target = os.listdir(self.target_root_dir)
        self.list_files_input = os.listdir(self.input_root_dir)

    def __len__(self):
        return len(self.list_files_target)

    def __getitem__(self, item):
        img_file_target = self.list_files_target[item]
        img_path_target = os.path.join(self.target_root_dir, img_file_target)
        image_target = np.array(Image.open(img_path_target))
        target_image = Image.fromarray(image_target)

        img_file_input = self.list_files_input[item]
        img_path_input = os.path.join(self.input_root_dir, img_file_input)
        image_input = np.array(Image.open(img_path_input))
        input_image = Image.fromarray(image_input)

        input_image = self.transform(input_image)
        target_image = self.transform(target_image)

        return input_image, target_image