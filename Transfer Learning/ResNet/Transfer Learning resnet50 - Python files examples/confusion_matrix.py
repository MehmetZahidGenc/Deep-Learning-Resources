import torchvision
from torch.utils.data.dataloader import DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

dataset_path = "your_dataset_path" # dataset for testing

batch_size = 64

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ]),
}

dataset = torchvision.datasets.ImageFolder(dataset_path, transform=data_transforms["validation"])

testloader = DataLoader(dataset, batch_size*2)

classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'] # change classes for your situation
model = torch.load('model_tl_pytorch2.pth') # model path 


y_pred = []
y_true = []

# iterate over test data
for inputs, labels in testloader:
    output = model(inputs)  # Feed Network

    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    y_pred.extend(output)  # Save Prediction

    labels = labels.data.cpu().numpy()
    y_true.extend(labels)  # Save Truth

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 100, index=[i for i in classes],
                     columns=[i for i in classes])
plt.figure(figsize=(12, 7))
sn.heatmap(df_cm, annot=True)

plt.show()
