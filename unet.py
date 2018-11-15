"""U-Net with attention gates."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_gate = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.W_skip = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=False), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        if g.shape[2:] != s.shape[2:]:
            g = F.interpolate(g, size=s.shape[2:])
        return skip * self.psi(self.relu(g + s))

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, features=(64, 128, 256, 512)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = in_channels
        for f in features:
            self.encoders.append(DoubleConv(prev, f))
            self.pools.append(nn.MaxPool2d(2))
            prev = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.attns.append(AttentionGate(f, f, f // 2))
            self.decoders.append(DoubleConv(f * 2, f))
        self.final = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for up, attn, dec, skip in zip(self.upconvs, self.attns, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:])
            x = dec(torch.cat([x, attn(x, skip)], dim=1))
        return self.final(x)
