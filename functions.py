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


def train(cfg, data, model, tracker, track=True):
    batch_size = cfg["batch_size"]
    num_epochs = cfg["num_epochs"]

    model.train()

    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        train_epoch(cfg, model, train_loader, epoch)
        if track:
            track_general(cfg, model, train_loader, tracker)
            # track_nc1(cfg, model, train_loader, tracker)
            track_nc2(cfg, model, train_loader, tracker)
            print("Tracking complete!")


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



@torch.no_grad()
def track_general(cfg, model, loader, tracker):
    device = cfg["device"]

    class_data_len = [0 for _ in range(model.num_classes)]

    for _, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        layer0_output = model.get_layer0_output(X).reshape((X.shape[0], -1))
        layer1_output = model.get_layer1_output(X).reshape((X.shape[0], -1))
        layer2_output = model.get_layer2_output(X).reshape((X.shape[0], -1))
        layer3_output = model.get_layer3_output(X).reshape((X.shape[0], -1))
        layer4_output = model.get_layer4_output(X).reshape((X.shape[0], -1))
        penult_output = model.get_penult_output(X).reshape((X.shape[0], -1))
        
        for c in range(model.num_classes):
            indices = (y == c).nonzero(as_tuple=True)[0]
            class_data_len[c] += len(indices)
            # If no class-c data in this batch
            if len(indices) == 0:
                if str(c) not in tracker.layer0_means:
                    tracker.layer0_means[str(c)] = torch.zeros(layer0_output.shape[1]).to(device)
                    tracker.layer1_means[str(c)] = torch.zeros(layer1_output.shape[1]).to(device)
                    tracker.layer2_means[str(c)] = torch.zeros(layer2_output.shape[1]).to(device)
                    tracker.layer3_means[str(c)] = torch.zeros(layer3_output.shape[1]).to(device)
                    tracker.layer4_means[str(c)] = torch.zeros(layer4_output.shape[1]).to(device)
                    tracker.penult_means[str(c)] = torch.zeros(penult_output.shape[1]).to(device)

            # Otherwise, accumulate class means
            layer0_output_c = layer0_output[indices]
            layer1_output_c = layer1_output[indices]
            layer2_output_c = layer2_output[indices]
            layer3_output_c = layer3_output[indices]
            layer4_output_c = layer4_output[indices]
            penult_output_c = penult_output[indices]

            if str(c) not in tracker.layer0_means:
                tracker.layer0_means[str(c)] = torch.sum(layer0_output_c, dim=0).to(device)
                tracker.layer1_means[str(c)] = torch.sum(layer1_output_c, dim=0).to(device)
                tracker.layer2_means[str(c)] = torch.sum(layer2_output_c, dim=0).to(device)
                tracker.layer3_means[str(c)] = torch.sum(layer3_output_c, dim=0).to(device)
                tracker.layer4_means[str(c)] = torch.sum(layer4_output_c, dim=0).to(device)
                tracker.penult_means[str(c)] = torch.sum(penult_output_c, dim=0).to(device)
            else:
                tracker.layer0_means[str(c)] += torch.sum(layer0_output_c, dim=0).to(device)
                tracker.layer1_means[str(c)] += torch.sum(layer1_output_c, dim=0).to(device)
                tracker.layer2_means[str(c)] += torch.sum(layer2_output_c, dim=0).to(device)
                tracker.layer3_means[str(c)] += torch.sum(layer3_output_c, dim=0).to(device)
                tracker.layer4_means[str(c)] += torch.sum(layer4_output_c, dim=0).to(device)
                tracker.penult_means[str(c)] += torch.sum(penult_output_c, dim=0).to(device)
            
    # After the loop, scale the class means; also compute the global mean for each layer
    for c in range(model.num_classes):
        tracker.layer0_means[str(c)] /= class_data_len[c]
        tracker.layer1_means[str(c)] /= class_data_len[c]
        tracker.layer2_means[str(c)] /= class_data_len[c]
        tracker.layer3_means[str(c)] /= class_data_len[c]
        tracker.layer4_means[str(c)] /= class_data_len[c]
        tracker.penult_means[str(c)] /= class_data_len[c]

        if c == 0:
            tracker.layer0_gmean = tracker.layer0_means[str(c)]
            tracker.layer1_gmean = tracker.layer1_means[str(c)]
            tracker.layer2_gmean = tracker.layer2_means[str(c)]
            tracker.layer3_gmean = tracker.layer3_means[str(c)]
            tracker.layer4_gmean = tracker.layer4_means[str(c)]
            tracker.penult_gmean = tracker.penult_means[str(c)]
        else:
            tracker.layer0_gmean += tracker.layer0_means[str(c)]
            tracker.layer1_gmean += tracker.layer1_means[str(c)]
            tracker.layer2_gmean += tracker.layer2_means[str(c)]
            tracker.layer3_gmean += tracker.layer3_means[str(c)]
            tracker.layer4_gmean += tracker.layer4_means[str(c)]
            tracker.penult_gmean += tracker.penult_means[str(c)]

    # Scale the global mean
    tracker.layer0_gmean /= model.num_classes
    tracker.layer1_gmean /= model.num_classes
    tracker.layer2_gmean /= model.num_classes
    tracker.layer3_gmean /= model.num_classes
    tracker.layer4_gmean /= model.num_classes
    tracker.penult_gmean /= model.num_classes

    # For each layer, compute distances from class means to global mean
    for c in range(model.num_classes):
        tracker.layer0_class_dists[str(c)] = tracker.layer0_means[str(c)] - tracker.layer0_gmean
        tracker.layer1_class_dists[str(c)] = tracker.layer1_means[str(c)] - tracker.layer1_gmean
        tracker.layer2_class_dists[str(c)] = tracker.layer2_means[str(c)] - tracker.layer2_gmean
        tracker.layer3_class_dists[str(c)] = tracker.layer3_means[str(c)] - tracker.layer3_gmean
        tracker.layer4_class_dists[str(c)] = tracker.layer4_means[str(c)] - tracker.layer4_gmean
        tracker.penult_class_dists[str(c)] = tracker.penult_means[str(c)] - tracker.penult_gmean


@torch.no_grad() # FIXME: Incorrect variance computations?
def track_nc1(cfg, model, loader, tracker):
    device = cfg["device"]

    class_data_len = [0 for _ in range(model.num_classes)]
    
    # Compute the within-class variance
    for _, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        layer0_output = model.get_layer0_output(X).reshape((X.shape[0], -1))
        layer1_output = model.get_layer1_output(X).reshape((X.shape[0], -1))
        layer2_output = model.get_layer2_output(X).reshape((X.shape[0], -1))
        layer3_output = model.get_layer3_output(X).reshape((X.shape[0], -1))
        layer4_output = model.get_layer4_output(X).reshape((X.shape[0], -1))
        penult_output = model.get_penult_output(X).reshape((X.shape[0], -1))

        for c in range(model.num_classes):
            indices = (y == c).nonzero(as_tuple=True)[0]
            # If no class-c data in this batch
            if len(indices) == 0:
                if str(c) not in tracker.var_layer0:
                    tracker.var_layer0[str(c)] = 0
                    tracker.var_layer1[str(c)] = 0
                    tracker.var_layer2[str(c)] = 0
                    tracker.var_layer3[str(c)] = 0
                    tracker.var_layer4[str(c)] = 0
                    tracker.var_penult[str(c)] = 0
            
            # Otherwise, accumulate class variances
            layer0_output_c = layer0_output[indices]
            layer1_output_c = layer1_output[indices]
            layer2_output_c = layer2_output[indices]
            layer3_output_c = layer3_output[indices]
            layer4_output_c = layer4_output[indices]
            penult_output_c = penult_output[indices]

            if str(c) not in tracker.var_layer0:
                tracker.var_layer0[str(c)] = torch.sum(torch.square(layer0_output_c - tracker.layer0_means[str(c)])).cpu().numpy()
                tracker.var_layer1[str(c)] = torch.sum(torch.square(layer1_output_c - tracker.layer1_means[str(c)])).cpu().numpy()
                tracker.var_layer2[str(c)] = torch.sum(torch.square(layer2_output_c - tracker.layer2_means[str(c)])).cpu().numpy()
                tracker.var_layer3[str(c)] = torch.sum(torch.square(layer3_output_c - tracker.layer3_means[str(c)])).cpu().numpy()
                tracker.var_layer4[str(c)] = torch.sum(torch.square(layer4_output_c - tracker.layer4_means[str(c)])).cpu().numpy()
                tracker.var_penult[str(c)] = torch.sum(torch.square(penult_output_c - tracker.penult_means[str(c)])).cpu().numpy()
            else:
                tracker.var_layer0[str(c)] += torch.sum(torch.square(layer0_output_c - tracker.layer0_means[str(c)])).cpu().numpy()
                tracker.var_layer1[str(c)] += torch.sum(torch.square(layer1_output_c - tracker.layer1_means[str(c)])).cpu().numpy()
                tracker.var_layer2[str(c)] += torch.sum(torch.square(layer2_output_c - tracker.layer2_means[str(c)])).cpu().numpy()
                tracker.var_layer3[str(c)] += torch.sum(torch.square(layer3_output_c - tracker.layer3_means[str(c)])).cpu().numpy()
                tracker.var_layer4[str(c)] += torch.sum(torch.square(layer4_output_c - tracker.layer4_means[str(c)])).cpu().numpy()
                tracker.var_penult[str(c)] += torch.sum(torch.square(penult_output_c - tracker.penult_means[str(c)])).cpu().numpy()

    # After the loop, scale the class variances and compute the NC1 criterion
    for c in range(model.num_classes):
        tracker.var_layer0[str(c)] /= class_data_len[c]
        tracker.var_layer1[str(c)] /= class_data_len[c]
        tracker.var_layer2[str(c)] /= class_data_len[c]
        tracker.var_layer3[str(c)] /= class_data_len[c]
        tracker.var_layer4[str(c)] /= class_data_len[c]
        tracker.var_penult[str(c)] /= class_data_len[c]

    tracker.nc1_layer0.append(sum(tracker.var_layer0.values()))
    tracker.nc1_layer1.append(sum(tracker.var_layer1.values()))
    tracker.nc1_layer2.append(sum(tracker.var_layer2.values()))
    tracker.nc1_layer3.append(sum(tracker.var_layer3.values()))
    tracker.nc1_layer4.append(sum(tracker.var_layer4.values()))
    tracker.nc1_penult.append(sum(tracker.var_penult.values()))


@torch.no_grad()
def track_nc2(cfg, model, loader, tracker):
    device = cfg["device"]

    # NC2: Equinorm
    for c in range(model.num_classes):
        tracker.layer0_norms[c] = torch.norm(tracker.layer0_means[str(c)] -  tracker.layer0_gmean)
        tracker.layer1_norms[c] = torch.norm(tracker.layer1_means[str(c)] -  tracker.layer1_gmean)
        tracker.layer2_norms[c] = torch.norm(tracker.layer2_means[str(c)] -  tracker.layer2_gmean)
        tracker.layer3_norms[c] = torch.norm(tracker.layer3_means[str(c)] -  tracker.layer3_gmean)
        tracker.layer4_norms[c] = torch.norm(tracker.layer4_means[str(c)] -  tracker.layer4_gmean)
        tracker.penult_norms[c] = torch.norm(tracker.penult_means[str(c)] -  tracker.penult_gmean)

    tracker.nc2_eqn_layer0.append(torch.std(tracker.layer0_norms) / torch.mean(tracker.layer0_norms))
    tracker.nc2_eqn_layer1.append(torch.std(tracker.layer1_norms) / torch.mean(tracker.layer1_norms))
    tracker.nc2_eqn_layer2.append(torch.std(tracker.layer2_norms) / torch.mean(tracker.layer2_norms))
    tracker.nc2_eqn_layer3.append(torch.std(tracker.layer3_norms) / torch.mean(tracker.layer3_norms))
    tracker.nc2_eqn_layer4.append(torch.std(tracker.layer4_norms) / torch.mean(tracker.layer4_norms))
    tracker.nc2_eqn_penult.append(torch.std(tracker.penult_norms) / torch.mean(tracker.penult_norms))

    # NC2: Equiangularity
    for c1 in range(model.num_classes):
        for c2 in range(model.num_classes):
            tracker.layer0_cosmap[c1, c2] = torch.dot(tracker.layer0_class_dists[str(c1)], tracker.layer0_class_dists[str(c2)]) / (tracker.layer0_norms[c1] * tracker.layer0_norms[c2])
            tracker.layer1_cosmap[c1, c2] = torch.dot(tracker.layer1_class_dists[str(c1)], tracker.layer1_class_dists[str(c2)]) / (tracker.layer1_norms[c1] * tracker.layer1_norms[c2])
            tracker.layer2_cosmap[c1, c2] = torch.dot(tracker.layer2_class_dists[str(c1)], tracker.layer2_class_dists[str(c2)]) / (tracker.layer2_norms[c1] * tracker.layer2_norms[c2])
            tracker.layer3_cosmap[c1, c2] = torch.dot(tracker.layer3_class_dists[str(c1)], tracker.layer3_class_dists[str(c2)]) / (tracker.layer3_norms[c1] * tracker.layer3_norms[c2])
            tracker.layer4_cosmap[c1, c2] = torch.dot(tracker.layer4_class_dists[str(c1)], tracker.layer4_class_dists[str(c2)]) / (tracker.layer4_norms[c1] * tracker.layer4_norms[c2])
            tracker.penult_cosmap[c1, c2] = torch.dot(tracker.penult_class_dists[str(c1)], tracker.penult_class_dists[str(c2)]) / (tracker.penult_norms[c1] * tracker.penult_norms[c2])

    max_angle_const = 1 / (model.num_classes - 1)

    tracker.layer0_cosmap += max_angle_const
    tracker.layer1_cosmap += max_angle_const
    tracker.layer2_cosmap += max_angle_const
    tracker.layer3_cosmap += max_angle_const
    tracker.layer4_cosmap += max_angle_const
    tracker.penult_cosmap += max_angle_const

    tracker.nc2_eqa_layer0.append(torch.mean(tracker.layer0_cosmap))
    tracker.nc2_eqa_layer1.append(torch.mean(tracker.layer1_cosmap))
    tracker.nc2_eqa_layer2.append(torch.mean(tracker.layer2_cosmap))
    tracker.nc2_eqa_layer3.append(torch.mean(tracker.layer3_cosmap))
    tracker.nc2_eqa_layer4.append(torch.mean(tracker.layer4_cosmap))
    tracker.nc2_eqa_penult.append(torch.mean(tracker.penult_cosmap))
