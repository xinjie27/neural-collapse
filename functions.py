import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from graphs import *


def get_dataset(cfg):
    dataset = cfg["dataset"]
    train = True if cfg["mode"] == "train" else False

    if dataset == "mnist":
        data = datasets.MNIST(root="data", train=train, download=True, transform=ToTensor())
        num_classes = 10
        in_channels = 1
    else:
        # TODO
        data = None
        num_classes = None

    return data, in_channels, num_classes

g = graphs()
def train(data, model, cfg, g):
    device = cfg["device"]
    batch_size = cfg["batch_size"]
    learning_rate = cfg["learning_rate"]
    num_epochs = cfg["n_epochs"]

    model.train()

    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Perform analysis
            


def analysis(data, model): 

    return None