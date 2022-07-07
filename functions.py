import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from ResNet import ResNetTracker


def get_dataset():
    data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
    num_classes = 10
    in_channels = 1

    return data, in_channels, num_classes


def train(cfg, data, model, tracker):
    batch_size = cfg["batch_size"]
    num_epochs = cfg["num_epochs"]

    model.train()

    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        train_epoch(cfg, model, train_loader, epoch)
        track_nc1(cfg, model, train_loader, tracker, len(data))


def train_epoch(cfg, model, loader, epoch):
    device = cfg["device"]
    learning_rate = cfg["learning_rate"]

    avg_accuracy = 0
    avg_loss = 0
    for batch_idx, (X, y) in enumerate(loader):
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

        batch_accuracy = torch.mean((torch.argmax(pred, dim=1) == y).float()).item()
        avg_accuracy += batch_accuracy
        avg_loss += loss
    
    avg_accuracy /= (batch_idx + 1)
    avg_loss /= (batch_idx + 1)
    print(f"Epoch {epoch + 1}: Loss {avg_loss:.4f}, Accuracy {avg_accuracy:.4f}")


def track_nc1(cfg, model, loader, tracker, data_size):
    device = cfg["device"]
    batch_size = cfg["batch_size"]

    for _, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        model_output = model(X).reshape((batch_size, -1))
        
        for c in range(model.num_classes):
            indices = (y == c).nonzero(as_tuple=True)[0]
            # If no class-c data in this batch
            if len(indices) == 0:
                if str(c) not in tracker.output_means:
                    tracker.output_means[str(c)] = torch.zeros_like(model_output.shape[1])
                else:
                    tracker.output_means[str(c)] += torch.zeros_like(model_output.shape[1])
            
            # Otherwise, accumulate class means
            model_output_c = model_output[indices]
            if str(c) not in tracker.output_means:
                tracker.output_means[str(c)] = torch.sum(model_output, dim=0)
            else:
                tracker.output_means[str(c)] += torch.sum(model_output, dim=0)

            
    # After the loop, scale the class means
    for c in range(model.num_classes):
        tracker.output_means[str(c)] /= data_size
    
    # Compute the within-class variance
    for _, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        model_output = model(X).reshape((batch_size, -1))

        for c in range(model.num_classes):
            indices = (y == c).nonzero(as_tuple=True)[0]
            # If no class-c data in this batch
            if len(indices) == 0:
                if str(c) not in tracker.var_output:
                    tracker.var_output[str(c)] = torch.zeros_like(model_output.shape[1])
                else:
                    tracker.var_output[str(c)] += torch.zeros_like(model_output.shape[1])
            
            # Otherwise, accumulate class variances
            model_output_c = model_output[indices]
            if str(c) not in tracker.var_output:
                tracker.var_output[str(c)] = torch.sum(torch.square(model_output_c - tracker.output_means[str(c)]))
            else:
                tracker.var_output[str(c)] += torch.sum(torch.square(model_output_c - tracker.output_means[str(c)]))

    # After the loop, scale the class variances and compute the NC1 criterion
    tracker.nc1_output.append(sum(tracker.var_output.values()) / data_size)
