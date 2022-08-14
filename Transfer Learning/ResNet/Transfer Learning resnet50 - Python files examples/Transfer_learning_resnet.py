import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch


resnet_50 = models.resnet50(pretrained=True)


for param in resnet_50.parameters():
    param.requires_grad = False


# just change fully connected layer to adjust for our own classes
resnet_50.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 5))

dataset_path = "your_data_set_path"

num_classes = 5 # daisy, dandelion, rose, sunflower, tulip

# Hyper parameters
num_epochs = 25
batch_size = 64
learning_rate = 0.0001

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


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


# we use just validation transform because our train and test dataset were combined in one folder
dataset = torchvision.datasets.ImageFolder(dataset_path, transform=data_transforms["validation"])

print(len(dataset))

test_size = int(len(dataset) * 0.2) # use %20 of data for validation or test
trains_size = len(dataset) - test_size
train_dataset, test_dataset = random_split(dataset, [trains_size, test_size])

trainloader = DataLoader(train_dataset, batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size*2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet_50.fc.parameters())


print(len(train_dataset), len(test_dataset))
print("\n")


def train_model(model, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in trainloader:

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(train_dataset)
                epoch_acc = running_corrects.double() / len(train_dataset)

                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc))
            else:
                model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in testloader:

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(test_dataset)
                epoch_acc = running_corrects.double() / len(test_dataset)

                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc))
                print("\n")
    return model


model_trained = train_model(resnet_50, criterion, optimizer, num_epochs=num_epochs)

FILE = "model_tl_pytorch.pth"
torch.save(model_trained, FILE)
