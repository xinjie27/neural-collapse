import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from loss import NCLoss
from ResNet import ResNetTracker
from track import *


def get_dataset():
    data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
    num_classes = 10
    in_channels = 1

    return data, in_channels, num_classes


def train(cfg, data, model, track=False):
    batch_size = cfg["batch_size"]
    num_epochs = cfg["num_epochs"]
    output_dir = cfg["output_dir"]

    model.train()

    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    if track:
        tracker = ResNetTracker(model.num_classes, num_epochs, output_dir)
        for epoch in range(num_epochs):
            train_epoch(cfg, model, train_loader, epoch, nn.CrossEntropyLoss())
            track_general(cfg, model, train_loader, tracker)
            # track_nc1(cfg, model, train_loader, tracker)
            track_nc2(cfg, model, train_loader, tracker)
            print("Tracking complete!")
        tracker.plot_nc2()
    else:
        losses = []
        accuracies = []
        for epoch in range(num_epochs):
            nc_loss = NCLoss(cfg)
            epoch_loss, epoch_accuracy = train_epoch(cfg, model, train_loader, epoch, nn.CrossEntropyLoss())
            losses.append(epoch_loss)
            accuracies.append(epoch_accuracy)

        # Plot loss and accuracy across epochs
        plt.figure(1)
        plt.plot(range(1, num_epochs + 1), losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(output_dir + "/loss.png")

        plt.figure(2)
        plt.plot(range(1, num_epochs + 1), accuracies)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.savefig(output_dir + "/accuracy.png")
        


def train_epoch(cfg, model, loader, epoch, loss_fn, print_acc=True):
    device = cfg["device"]
    learning_rate = cfg["learning_rate"]

    avg_accuracy = 0
    avg_loss = 0
    for batch_idx, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_accuracy = torch.mean((torch.argmax(pred, dim=1) == y).float()).item()
        avg_accuracy += batch_accuracy
        avg_loss += loss
    
    avg_accuracy /= (batch_idx + 1)
    avg_loss /= (batch_idx + 1)

    if print_acc:
        print(f"Epoch {epoch + 1}: Loss {avg_loss:.4f}, Accuracy {avg_accuracy:.4f}")
    else:
        print(f"Epoch {epoch + 1}: Loss {avg_loss:.4f}, Accuracy N/A")

    return avg_loss, avg_accuracy


def transfer_train(cfg, data, model, fc_epochs=10):
    batch_size = cfg["batch_size"]
    num_epochs = cfg["num_epochs"]
    output_dir = cfg["output_dir"]
    device = cfg["device"]

    model.train()
    
    # Remove the FC layer from the model
    model.gap = nn.Identity()
    model.fc = nn.Identity()

    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    losses = []
    accuracies = []
    for epoch in range(num_epochs):
        nc_loss = NCLoss(cfg)
        epoch_loss, epoch_accuracy = train_epoch(cfg, model, train_loader, epoch, nn.CrossEntropyLoss(), print_acc=False) # nc_loss.loss
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

    print("FC Layer Training.")

    # Freeze the model weights and only train the final FC layer
    for param in model.parameters():
        param.requires_grad = False
    
    model.gap = torch.nn.AdaptiveAvgPool2d(1).to(device)
    model.gap.requires_grad = True
    model.fc = torch.nn.Linear(model.num_features, model.num_classes).to(device)
    model.fc.requires_grad = True
    
    fc_losses = []
    for epoch in range(fc_epochs):
        epoch_loss, epoch_accuracy = train_epoch(cfg, model, train_loader, epoch, nn.CrossEntropyLoss())
        fc_losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
    
    plt.figure(1)
    plt.plot(range(1, num_epochs + 1), losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(output_dir + "/nc_loss.png")

    plt.figure(2)
    plt.plot(range(1, num_epochs + fc_epochs + 1), accuracies)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(output_dir + "/accuracy.png")