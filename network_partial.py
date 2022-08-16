import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ResNet import ResBlock


class ResNet2Layer(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = nn.Sequential(
            ResBlock(64, 64, downsample=False), ResBlock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            ResBlock(64, 128, downsample=True), ResBlock(128, 128, downsample=False)
        )

        self.num_features = 128
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(128, num_classes)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.gap(input)
        input = torch.flatten(input, 1)
        input = self.fc(input)

        return input


class ResNetFC(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.fc = torch.nn.Linear(in_channels, num_classes)

    def forward(self, input):
        input = torch.flatten(input, 1)
        return self.fc(input)