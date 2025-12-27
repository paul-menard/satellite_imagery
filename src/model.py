import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision as tv
import sklearn.model_selection
from torch.utils.data import DataLoader, random_split

stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
transform = tv.transforms.Compose([
    tv.transforms.RandomHorizontalFlip(p=0.5),
    tv.transforms.RandomVerticalFlip(p=0.5),   
    tv.transforms.RandomRotation(degrees=15),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(*stats)
])

path = "../data"
dataset = tv.datasets.EuroSAT(path, download = True, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle = True)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False)

class Satelite(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer3 =  nn.Linear(in_features=16*16*32, out_features=10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.layer3(x)
        return x
    
model = Satelite()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
loss = nn.CrossEntropyLoss()

print("\n Model created ! Training is starting")

for epoch in range(20):
    model.train()
    mean_loss = 0
    # i need to split all of this into batches
    for batch in train_loader:
        images, labels = batch
        labels_hat = model.forward(images).squeeze()
        lost = loss(labels_hat, labels)

        optimizer.zero_grad()
        lost.backward()

        optimizer.step()

        mean_loss += lost
    mean_loss = mean_loss/338
    print(f'Epoch n°{epoch+1} | loss = {mean_loss}')

print("\n Début du Test...")
model.eval() 

correct_predictions = 0
total_predictions = 0

with torch.no_grad(): 
    for images, labels in test_loader:
        outputs = model(images)
    
        _, predicted = torch.max(outputs, 1)
        
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

accuracy = (correct_predictions / total_predictions) * 100
print(f" Précision Finale : {accuracy:.2f}%")
