from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResDoubleConv(nn.Module):
    """ BN -> ReLU -> Conv3D -> BN -> ReLU -> Conv3D """

    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.skip = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        )

    def forward(self, x):
        return self.double_conv(x) + self.skip(x)


class ResDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            ResDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class ResUp(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=False):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = ResDoubleConv(in_channels + in_channels // 2, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)
            self.conv = ResDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResUNet3d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.input_layer = nn.Sequential(
            nn.Conv3d(in_channels, n_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=n_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        )
        self.input_skip = nn.Conv3d(in_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.enc1 = ResDown(n_channels, 2 * n_channels)
        self.enc2 = ResDown(2 * n_channels, 4 * n_channels)
        self.enc3 = ResDown(4 * n_channels, 8 * n_channels)
        self.bridge = ResDown(8 * n_channels, 16 * n_channels)
        self.dec1 = ResUp(16 * n_channels, 8 * n_channels)
        self.dec2 = ResUp(8 * n_channels, 4 * n_channels)
        self.dec3 = ResUp(4 * n_channels, 2 * n_channels)
        self.dec4 = ResUp(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)
        # x1:n -> x2:2n
        x2 = self.enc1(x1)
        # x2:2n -> x3:4n
        x3 = self.enc2(x2)
        # x3:4n -> x4:8n
        x4 = self.enc3(x3)
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
    device = torch.device("cpu")
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    model = ResUNet3d(4, 3, n_channels=32).to(device)
    # batch, channel, x, y, z
    x = torch.rand(4, 4, 64, 96, 96)
    x = x.to(device)
    model.forward(x)
    print("everything ok!")


if __name__ == '__main__':
    main()
