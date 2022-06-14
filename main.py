import numpy as np
import torch

from models.ResNet import ResNet18


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    resnet_model = ResNet18(3, in_channels=[2, 2, 2, 2], outputs=10)
    resnet_model.to(device)