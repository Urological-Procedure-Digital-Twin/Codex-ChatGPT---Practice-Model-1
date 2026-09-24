"""A small U-Net: three downsampling stages, skip connections, one output logit.

Input:  N x 3 x 256 x 256 normalized RGB images.
Output: N x 1 x 256 x 256 logits (apply sigmoid to obtain probabilities).
"""

import torch
from torch import nn


class DoubleConv(nn.Sequential):
    """Two spatial convolutions; GroupNorm works even with small batches."""

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True),
        )


class SmallUNet(nn.Module):
    """Channels 8 -> 16 -> 32 -> 64, then back to 8; trained from scratch."""

    def __init__(self, base_channels=8):
        super().__init__()
        b = base_channels
        self.enc1 = DoubleConv(3, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.pool = nn.MaxPool2d(2)
        self.bridge = DoubleConv(b * 4, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)
        self.output = nn.Conv2d(b, 1, 1)

    def forward(self, image):
        e1 = self.enc1(image)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        center = self.bridge(self.pool(e3))
        d3 = self.dec3(torch.cat((self.up3(center), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        return self.output(d1)
