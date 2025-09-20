import torch
import torch.nn as nn
from typing import List


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch*2, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2); dx = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, enc_layers: List[nn.Module], chs: List[int]):
        super().__init__()
        self.enc = nn.ModuleList(enc_layers)
        self.pool = nn.MaxPool2d(2)
        self.center = nn.Sequential(nn.Conv2d(chs[-1], chs[-1], 3, padding=1), nn.ReLU(True))
        self.up4 = Up(chs[-1], 256)
        self.up3 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up1 = Up(64, 64)
        self.outc = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        feats = []
        h = x
        for i, l in enumerate(self.enc):
            h = l(h) if i == 0 else l(self.pool(h))
            feats.append(h)
        h = self.center(feats[-1])
        h = self.up4(h, feats[-1])
        h = self.up3(h, feats[-2])
        h = self.up2(h, feats[-3])
        h = self.up1(h, feats[-4])
        logits = self.outc(h)
        return logits
