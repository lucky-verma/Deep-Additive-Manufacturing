import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True))
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels,
                                     in_channels // 2,
                                     kernel_size=2,
                                     stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x, enc):
        x = self.up(x)
        diffZ = enc.size()[2] - x.size()[2]
        diffY = enc.size()[3] - x.size()[3]
        diffX = enc.size()[4] - x.size()[4]
        x = F.pad(
            x,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([enc, x], dim=1)
        x = self.conv(x)
        return x
class PredNet(nn.Module):

    def __init__(self):
        super(PredNet, self).__init__()
        self.enc1 = EncoderBlock(1, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        self.dec1 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128, 64)
        self.out = nn.Sequential(nn.Conv3d(64, 1, kernel_size=1),
                                 nn.BatchNorm3d(1), nn.Sigmoid())

    def forward(self, x):
        x = F.interpolate(x, size=(32, 32, 32))
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        dec1 = self.dec1(enc4, enc3)
        dec2 = self.dec2(dec1, enc2)
        dec3 = self.dec3(dec2, enc1)
        out = self.out(dec3)
        out = F.interpolate(out, size=(32, 32, 32))
        return out.view(1, 1, 32, 32, 32)

