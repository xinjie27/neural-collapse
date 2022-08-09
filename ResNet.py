import numpy as np
from matplotlib import pyplot as plt
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

    def get_penult_output(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input, 1)
        return input


class ResNetTracker:
    def __init__(self, num_classes, num_epochs, output_dir):
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.output_dir = output_dir

        # General indicators
        # Intermediate layer class means; dictionaries of (class_idx: 1D class mean vector)
        self.layer0_means = {}
        self.layer1_means = {}
        self.layer2_means = {}
        self.layer3_means = {}
        self.layer4_means = {}
        self.penult_means = {}
        # Global means
        self.layer0_gmean = None
        self.layer1_gmean = None
        self.layer2_gmean = None
        self.layer3_gmean = None
        self.layer4_gmean = None
        self.penult_gmean = None
        # Distances from class means to global means
        self.layer0_class_dists = {}
        self.layer1_class_dists = {}
        self.layer2_class_dists = {}
        self.layer3_class_dists = {}
        self.layer4_class_dists = {}
        self.penult_class_dists = {}

        # NC1
        # Intermediate layer variances; dictionaries of (class_idx: 1D class variance vector)
        self.var_layer0 = {}
        self.var_layer1 = {}
        self.var_layer2 = {}
        self.var_layer3 = {}
        self.var_layer4 = {}
        self.var_penult = {}
        # Intermediate layer NC1 criteria, each of length num_epochs
        self.nc1_layer0 = []
        self.nc1_layer1 = []
        self.nc1_layer2 = []
        self.nc1_layer3 = []
        self.nc1_layer4 = []
        self.nc1_penult = []

        # NC2
        # Norms of (class mean - global mean)
        self.layer0_norms = torch.zeros(self.num_classes)
        self.layer1_norms = torch.zeros(self.num_classes)
        self.layer2_norms = torch.zeros(self.num_classes)
        self.layer3_norms = torch.zeros(self.num_classes)
        self.layer4_norms = torch.zeros(self.num_classes)
        self.penult_norms = torch.zeros(self.num_classes)

        self.nc2_eqn_layer0 = []
        self.nc2_eqn_layer1 = []
        self.nc2_eqn_layer2 = []
        self.nc2_eqn_layer3 = []
        self.nc2_eqn_layer4 = []
        self.nc2_eqn_penult = []

        self.layer0_cosmap = torch.zeros([self.num_classes, self.num_classes])
        self.layer1_cosmap = torch.zeros([self.num_classes, self.num_classes])
        self.layer2_cosmap = torch.zeros([self.num_classes, self.num_classes])
        self.layer3_cosmap = torch.zeros([self.num_classes, self.num_classes])
        self.layer4_cosmap = torch.zeros([self.num_classes, self.num_classes])
        self.penult_cosmap = torch.zeros([self.num_classes, self.num_classes])

        self.nc2_eqa_layer0 = []
        self.nc2_eqa_layer1 = []
        self.nc2_eqa_layer2 = []
        self.nc2_eqa_layer3 = []
        self.nc2_eqa_layer4 = []
        self.nc2_eqa_penult = []

    def plot_nc1(self):
        plt.figure(1)
        plt.plot(range(1, self.num_epochs + 1), self.nc1_layer0)
        plt.xlabel("Epoch")
        plt.ylabel("Sum of in-class variances")
        plt.title("NC1: Layer 0")
        plt.savefig(self.output_dir + "/nc1_layer0.png")

        plt.figure(2)
        plt.plot(range(1, self.num_epochs + 1), self.nc1_layer1)
        plt.xlabel("Epoch")
        plt.ylabel("Sum of in-class variances")
        plt.title("NC1: Layer 1")
        plt.savefig(self.output_dir + "/nc1_layer1.png")

        plt.figure(3)
        plt.plot(range(1, self.num_epochs + 1), self.nc1_layer2)
        plt.xlabel("Epoch")
        plt.ylabel("Sum of in-class variances")
        plt.title("NC1: Layer 2")
        plt.savefig(self.output_dir + "/nc1_layer2.png")

        plt.figure(4)
        plt.plot(range(1, self.num_epochs + 1), self.nc1_layer3)
        plt.xlabel("Epoch")
        plt.ylabel("Sum of in-class variances")
        plt.title("NC1: Layer 3")
        plt.savefig(self.output_dir + "/nc1_layer3.png")

        plt.figure(5)
        plt.plot(range(1, self.num_epochs + 1), self.nc1_layer4)
        plt.xlabel("Epoch")
        plt.ylabel("Sum of in-class variances")
        plt.title("NC1: Layer 4")
        plt.savefig(self.output_dir + "/nc1_layer4.png")

        plt.figure(6)
        plt.plot(range(1, self.num_epochs + 1), self.nc1_penult)
        plt.xlabel("Epoch")
        plt.ylabel("Sum of in-class variances")
        plt.title("NC1: Penultimate Layer Features")
        plt.savefig(self.output_dir + "/nc1_penult.png")

    def _plot_nc2_equinorm(self):
        plt.figure(1)
        plt.plot(range(1, self.num_epochs + 1), self.nc2_eqn_layer0)
        plt.xlabel("Epoch")
        plt.ylabel("Std/Avg")
        plt.title("NC2: Layer 0 Equinorm")
        plt.savefig(self.output_dir + "/nc2_layer0_equinorm.png")

        plt.figure(2)
        plt.plot(range(1, self.num_epochs + 1), self.nc2_eqn_layer1)
        plt.xlabel("Epoch")
        plt.ylabel("Std/Avg")
        plt.title("NC2: Layer 1 Equinorm")
        plt.savefig(self.output_dir + "/nc2_layer1_equinorm.png")

        plt.figure(3)
        plt.plot(range(1, self.num_epochs + 1), self.nc2_eqn_layer2)
        plt.xlabel("Epoch")
        plt.ylabel("Std/Avg")
        plt.title("NC2: Layer 2 Equinorm")
        plt.savefig(self.output_dir + "/nc2_layer2_equinorm.png")

        plt.figure(4)
        plt.plot(range(1, self.num_epochs + 1), self.nc2_eqn_layer3)
        plt.xlabel("Epoch")
        plt.ylabel("Std/Avg")
        plt.title("NC2: Layer 3 Equinorm")
        plt.savefig(self.output_dir + "/nc2_layer3_equinorm.png")

        plt.figure(5)
        plt.plot(range(1, self.num_epochs + 1), self.nc2_eqn_layer4)
        plt.xlabel("Epoch")
        plt.ylabel("Std/Avg")
        plt.title("NC2: Layer 4 Equinorm")
        plt.savefig(self.output_dir + "/nc2_layer0_equinorm.png")

        plt.figure(6)
        plt.plot(range(1, self.num_epochs + 1), self.nc2_eqn_penult)
        plt.xlabel("Epoch")
        plt.ylabel("Std/Avg")
        plt.title("NC2: Penultimate Layer Equinorm")
        plt.savefig(self.output_dir + "/nc2_penult_equinorm.png")

    def _plot_nc2_equiangularity(self):
        plt.figure(1)
        plt.plot(range(1, self.num_epochs + 1), self.nc2_eqa_layer0)
        plt.xlabel("Epoch")
        plt.ylabel("Avg(|Shifted Cos|)")
        plt.title("NC2: Layer 0 Equiangularity")
        plt.savefig(self.output_dir + "/nc2_layer0_equiangularity.png")

        plt.figure(2)
        plt.plot(range(1, self.num_epochs + 1), self.nc2_eqa_layer1)
        plt.xlabel("Epoch")
        plt.ylabel("Avg(|Shifted Cos|)")
        plt.title("NC2: Layer 1 Equiangularity")
        plt.savefig(self.output_dir + "/nc2_layer1_equiangularity.png")

        plt.figure(3)
        plt.plot(range(1, self.num_epochs + 1), self.nc2_eqa_layer2)
        plt.xlabel("Epoch")
        plt.ylabel("Avg(|Shifted Cos|)")
        plt.title("NC2: Layer 2 Equiangularity")
        plt.savefig(self.output_dir + "/nc2_layer2_equiangularity.png")

        plt.figure(4)
        plt.plot(range(1, self.num_epochs + 1), self.nc2_eqa_layer3)
        plt.xlabel("Epoch")
        plt.ylabel("Avg(|Shifted Cos|)")
        plt.title("NC2: Layer 3 Equiangularity")
        plt.savefig(self.output_dir + "/nc2_layer3_equiangularity.png")

        plt.figure(5)
        plt.plot(range(1, self.num_epochs + 1), self.nc2_eqa_layer4)
        plt.xlabel("Epoch")
        plt.ylabel("Avg(|Shifted Cos|)")
        plt.title("NC2: Layer 4 Equiangularity")
        plt.savefig(self.output_dir + "/nc2_layer4_equiangularity.png")

        plt.figure(6)
        plt.plot(range(1, self.num_epochs + 1), self.nc2_eqa_penult)
        plt.xlabel("Epoch")
        plt.ylabel("Avg(|Shifted Cos|)")
        plt.title("NC2: Penultimate Layer Equiangularity")
        plt.savefig(self.output_dir + "/nc2_penult_equiangularity.png")

    
    def plot_nc2(self):
        self._plot_nc2_equinorm()
        self._plot_nc2_equiangularity()
