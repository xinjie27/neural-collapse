import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import models
import json

from functions import *
from ResNet import ResNet18, ResNetTracker
from network_partial import ResNet2Layer, ResNetFC


def get_config():
    """
    Parses the command-line arguments and sets configurations

    :return: config, a dictionary that stores all arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="outputs/",
        help="Output directory",
        dest="output_dir",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=350,
        help="Number of epochs",
        dest="num_epochs",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=256,
        help="Batch size",
        dest="batch_size",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
        dest="learning_rate",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=2,
        help="Lambda, regularizing coefficient",
        dest="lmbda",
    )

    args = parser.parse_args()
    config = {}

    for arg in vars(args):
        config[arg] = getattr(args, arg)

    return config


def save_config(cfg, filepath):
    with open(filepath, "w") as f:
        json.dump(cfg, f)
    

def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device
    output_dir = cfg["output_dir"]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    data, in_channels, num_classes = get_dataset()
    cfg["num_classes"] = num_classes

    # model = ResNet18(in_channels, num_classes)
    model = ResNet2Layer(in_channels, num_classes)
    # model = ResNetFC(784, num_classes)
    model.to(device)
    
    # train(cfg, data, model, track=False)
    transfer_train(cfg, data, model, fc_epochs=50)
    save_config(cfg, output_dir + "/config.json")
    


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
