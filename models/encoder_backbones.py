import torch.nn as nn
import torchvision.models as tv


def make_encoder(name: str = "resnet18", pretrained: bool = False):
    if name == "resnet18":
        m = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT if pretrained else None)
        layers = [nn.Sequential(m.conv1, m.bn1, m.relu), m.layer1, m.layer2, m.layer3, m.layer4]
        channels = [64, 64, 128, 256, 512]
        return layers, channels
    raise ValueError(f"Unknown encoder {name}")
