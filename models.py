import torch
from torch import nn
from torchvision import models


def build_resnet18(in_channels, num_classes):
    model = models.resnet18(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def build_vgg11(in_channels, num_classes):
    model = models.vgg11(pretrained=False, num_classes=num_classes)
    # TODO
    return model


def build_densenet():
    pass