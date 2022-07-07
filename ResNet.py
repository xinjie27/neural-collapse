import numpy as np
import torch
from torch import nn

"""
An implementation of ResNet18.
"""


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)


class ResNet18(nn.Module):
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

        self.layer3 = nn.Sequential(
            ResBlock(128, 256, downsample=True), ResBlock(256, 256, downsample=False)
        )

        self.layer4 = nn.Sequential(
            ResBlock(256, 512, downsample=True), ResBlock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, num_classes)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input, 1)
        input = self.fc(input)

        return input
    
    # Helper functions for analysis

    def get_layer0_output(self, input):
        input = self.layer0(input)
        return input

    def get_layer1_output(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        return input

    def get_layer2_output(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        return input

    def get_layer3_output(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        return input

    def get_layer4_output(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        return input


class ResNetTracker():
    def __init__(self):
        # Initialize class means
        self.layer0_means = {}
        self.layer1_means = {}
        self.layer2_means = {}
        self.layer3_means = {}
        self.layer4_means = {}
        self.output_means = {}

        self.var_layer0 = {}
        self.var_layer1 = {}
        self.var_layer2 = {}
        self.var_layer3 = {}
        self.var_layer4 = {}
        self.var_output = {}

        self.nc1_layer0 = []
        self.nc1_layer1 = []
        self.nc1_layer2 = []
        self.nc1_layer3 = []
        self.nc1_layer4 = []
        self.nc1_output = []