import torchvision
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


class confusionMatrix:
    def __init__(self, dataset_path, batch_size, transform, model_path, fig_filename, save_fig=False):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.transform = transform
        self.model_path = model_path
        self.save_fig = save_fig
        self.fig_filename = fig_filename
        self.model = torch.load(self.model_path)


    def createDataset(self):
        dataset = torchvision.datasets.ImageFolder(root=self.dataset_path, transform=self.transform)

        return dataset

    def createLoader(self, dataset):
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        return loader

    def class_names(self, dataset):
        dict = dataset.class_to_idx
        classes_list = list(dict.keys())

        return classes_list


    def make_CM(self):
        dataset = self.createDataset()
        loader = self.createLoader(dataset=dataset)
        classes = self.class_names(dataset=dataset)

        y_pred = []
        y_true = []

        # iterate over test data
        for inputs, labels in loader:
            output = self.model(inputs)

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 100, index=[i for i in classes],
                             columns=[i for i in classes])
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)

        if self.save_fig:
            plt.savefig(self.fig_filename + '.png')

        plt.show()