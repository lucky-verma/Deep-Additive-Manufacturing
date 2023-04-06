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


class SpatialBranch(nn.Module):

    def __init__(self):
        super(SpatialBranch, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        return x


class PredNet(nn.Module):

    def __init__(self):
        super(PredNet, self).__init__()
        self.enc1 = EncoderBlock(1, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        self.spatial = SpatialBranch()
        self.flatten = nn.Flatten()
        # step 4
        self.fc = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(1024, 4096),
                                nn.Sigmoid())

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # step 3
        enc4 = self.flatten(enc4)

        # output of the spatial
        spatial = self.spatial(x)
        spatial = self.flatten(spatial)

        # It is combined with the output of the spatial data branch
        x = torch.cat((enc4, spatial), dim=1)

        x = self.fc(x)
        x = x.view(-1, 16, 16, 16)

        return x
