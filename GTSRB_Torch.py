import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, transforms, Resize
from torchvision.io import read_image
from torch.utils.data import Dataset
import os
import pandas as pd
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import argparse

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="batch size", default=64, dest="batch_size")
    parser.add_argument("--lr", help="learning rate", default=1e-3, dest="lr")
    parser.add_argument("--epochs", help="number of epochs", default=20, dest="epochs")
    parser.add_argument("--model_path", help="path to save model", default="model.pth", dest="model_path")
    return parser.parse_args()

args = parseArguments()

# Print pytorch info
print(f"Using torch version {torch.__version__}")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} for training.")


# Transforms used on Data
transforms = transforms.Compose([Resize((30, 30)),
                                 ToTensor()])


# create data directory if not exists
os.makedirs("./data", exist_ok=True)

# Download Dataset
GTSRB_train = torchvision.datasets.GTSRB(
    root="./data",
    split="train",
    download=True,
    transform=transforms,
)

GTSRB_test = torchvision.datasets.GTSRB(
    root="./data",
    split="test",
    download=True,
    transform=transforms,
)

classes = range(43+1)


# Create data loaders
BATCH_SIZE = 64
trainloader = DataLoader(GTSRB_train, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(GTSRB_test, batch_size=BATCH_SIZE, shuffle=True)

for X,y in testloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    img = img.cpu()
    npimg = img.numpy()
    print(npimg.shape)
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

# Create model
class ConvelutionalNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 43),
        )
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

model = ConvelutionalNN().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/GTSRB_experiment_1')

dataiter = iter(trainloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)

writer.add_image('four_GTSRB_images', img_grid)

writer.add_graph(model, images)


running_loss = 0.0
EPOCHS = args.epochs
size = len(trainloader)
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}\n-------------------------------")
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            writer.add_scalar('training loss', running_loss / 100, epoch * len(trainloader) + i)
            loss, current = loss.item(), (i+1) * len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            running_loss = 0.0
    model.eval()
    size = len(testloader.dataset)
    num_batches = len(testloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for image,label in testloader:
            image,label = image.to(device), label.to(device)
            pred = model(image)
            test_loss += criterion(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    writer.add_scalar('accuracy', correct, epoch)
    print(f"\nTest Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
print("Finished Training")


torch.save(model.state_dict(), args.model_path)
print("Saved PyTorch Model State to model.pth")





