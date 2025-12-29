import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader, random_split

stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
transform = tv.transforms.Compose([
    tv.transforms.RandomHorizontalFlip(p=0.5),
    tv.transforms.RandomVerticalFlip(p=0.5),   
    tv.transforms.RandomRotation(degrees=15),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(*stats)
])


# 1. On récupère toutes les prédictions
all_preds = []
all_labels = []

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
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer4 =  nn.Linear(in_features=8*8*64, out_features=10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.layer4(x)
        return x

model = Satelite()
chemin_du_modele = "eurosat_cnn_92_33.pth"
state_dict = torch.load(chemin_du_modele)
model.load_state_dict(state_dict)

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        # Envoie sur GPU si dispo, sinon CPU
        # images = images.cuda() 
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# 2. On calcule la matrice
# Les lignes sont les VRAIES classes, les colonnes sont les PRÉDICTIONS
cm = confusion_matrix(all_labels, all_preds)

# 3. On l'affiche joliment
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=dataset.classes, 
            yticklabels=dataset.classes)
plt.xlabel('Prédiction')
plt.ylabel('Vraie Classe')
plt.title('Où le modèle se trompe-t-il ?')
plt.show()