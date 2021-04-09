from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=False):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        # x = torch.add(x2, x1)
        return self.conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.enc1 = DoubleConv(in_channels, n_channels)
        self.enc2 = Down(n_channels, 2 * n_channels)
        self.enc3 = Down(2 * n_channels, 4 * n_channels)
        self.enc4 = Down(4 * n_channels, 8 * n_channels)

        self.bridge = DoubleConv(8 * n_channels, 16 * n_channels)

        self.dec1 = Up(16 * n_channels, 8 * n_channels)
        self.dec2 = Up(8 * n_channels, 4 * n_channels)
        self.dec3 = Up(4 * n_channels, 2 * n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        # x:in -> x1:n
        x1 = self.enc1(x)
        # x1:n -> x2:2n
        x2 = self.enc2(x1)
        # x2:2n -> x3:4n
        x3 = self.enc3(x2)
        # x3:4n -> x4:8n
        x4 = self.enc4(x3)
        # x4:8n -> x5:16n
        bridge = self.bridge(x4)

        mask = self.dec1(bridge, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = UNet3d(4, 3, n_channels=32).to(device)
    # batch, channel, x, y, z
    x = torch.rand(4, 4, 64, 96, 96)
    x = x.to(device)
    model.forward(x)


if __name__ == '__main__':
    main()
