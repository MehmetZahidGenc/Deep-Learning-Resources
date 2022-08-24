import torch
import torch.nn as nn
from torchvision import datasets, transforms
from AE_model import AE

# parameters
nc = 1
num_epochs = 15
learning_rate = 1e-4
weight_decay = 1e-5

# transform
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(nc)], [0.5 for _ in range(nc)])
    ])

# We use MNIST Dataset
mnist_data = datasets.MNIST(root='dataset', train=True, download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=128, shuffle=True)

# Model
model = AE(in_channel=nc, features=[16, 32, 64])

if torch.cuda.is_available():
    model.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training
outputs = []
for epoch in range(num_epochs):
    for (mzg, _) in data_loader:
        rec = model(mzg)
        loss = criterion(rec, mzg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch:{epoch}, Loss:{loss.item():.4f}')
    outputs.append((epoch, mzg, rec))