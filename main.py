import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import models

from functions import *
from ResNet import ResNet18


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
    parser.add_argument(
        "--lambda",
        type=float,
        default=1,
        help="Lambda, regularizing coefficient",
        dest="lmbda",
    )

    args = parser.parse_args()
    config = {}

    for arg in vars(args):
        config[arg] = getattr(args, arg)

    return config


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device
    num_epochs = cfg["num_epochs"]
    output_dir = cfg["output_dir"]

    data, in_channels, num_classes = get_dataset()
    model = ResNet18(in_channels, num_classes)
    model.to(device)
    tracker = ResNetTracker(num_classes, num_epochs, output_dir)

    train(cfg, data, model, tracker)

    tracker.plot_nc2()


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
