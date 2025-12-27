import torchvision as tv

path = "../data"
dataset = tv.datasets.EuroSAT(path, download = True)

image, label = dataset[0]

print(image, label)