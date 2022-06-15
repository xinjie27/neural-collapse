import argparse
import os
import numpy as np
import torch
from torchvision import models

from functions import *
from models import *


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
    
    data, in_channels, num_classes = get_dataset(cfg)
    model = build_resnet18(in_channels, num_classes)
    model.to(device)

    train(data, model, cfg)
    pass



if __name__ == "__main__":
    cfg = get_config()
    main(cfg)