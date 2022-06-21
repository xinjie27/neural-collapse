import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import transforms

from functions import *
from models import *
from utils import *


def get_config():
    """
    Parses the command-line arguments and sets configurations

    :return: config, a dictionary that stores all arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["mnist"],
        required=True,
        help="Choose a dataset",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="outputs/",
        help="Output directory",
        dest="output_dir",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=5, help="Number of epochs", dest="n_epochs"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        help="Batch size",
        dest="batch_size",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Learning rate",
        dest="learning_rate",
    )

    args = parser.parse_args()
    config = {}

    for arg in vars(args):
        config[arg] = getattr(args, arg)

    return config


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device
    batch_size = cfg["batch_size"]
    learning_rate = cfg["learning_rate"]
    momentum = 0.9
    weight_decay = 5e-4
    data, in_channels, num_classes = get_dataset(cfg)

    # epoch_list  = [1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11,
    #             12,  13,  14,  16,  17,  19,  20,  22,  24,  27,   29,
    #             32,  35,  38,  42,  45,  50,  54,  59,  65,  71,   77,
    #             85,  92,  101, 110, 121, 132, 144, 158, 172, 188,  206,
    #             225, 245, 268, 293, 320, 350]
    epoch_list = [1, 2]
    epochs = epoch_list[-1]
    
    model = build_resnet18(in_channels, num_classes)
    model.to(device)
    # register hook that saves last-layer input into features
    classifier = model.fc
    classifier.register_forward_hook(hook)
    
    ## Will deal with this part later
    im_size             = 28
    padded_im_size      = 32
    C                   = 10
    input_ch            = 1
    transform = transforms.Compose([transforms.Pad((padded_im_size - im_size)//2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.1307,0.3081)])
    train_loader = DataLoader(data, train=True, batch_size=batch_size, shuffle=True, 
                            drop_last=True, transform=transform)
    analysis_loader = DataLoader(data, train=True, batch_size=batch_size, shuffle=True, 
                            drop_last=True, transform=transform)

    loss_name = "MSELoss"
    if loss_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
        criterion_summed = nn.CrossEntropyLoss(reduction='sum')
    elif loss_name == 'MSELoss':
        criterion = nn.MSELoss()
        criterion_summed = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, 
                            momentum=momentum, weight_decay=weight_decay)
    
    # See utils.py for more information
    # g is a graphs that stores data
    # S is a specs that stores specifications 
    # Can move more things in config to specs class
    g = graphs()
    S = specs()
    S.device = device
    S.optimizer = optimizer
    S.loss_name = loss_name
    S.criterion = criterion
    S.criterion_summed = criterion_summed
    S.epoch_list = epoch_list
    S.num_classes = C
    S.classifier = classifier
    S.weight_decay = weight_decay
    S.input_channels = input_ch

    # Begin epochs
    for epoch in range(1, epochs+1):
        train(model, cfg, train_loader, S)
        if epoch in epoch_list:
            analysis(g, model, analysis_loader, S, epoch)
    plotting(g, S)



if __name__ == "__main__":
    cfg = get_config()
    main(cfg)