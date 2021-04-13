from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int, num_groups=8):
        super(AttnBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=num_groups, num_channels=F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=num_groups, num_channels=F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        diffZ = x.size()[2] - g.size()[2]
        diffY = x.size()[3] - g.size()[3]
        diffX = x.size()[4] - g.size()[4]
        g = F.pad(g, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return torch.cat((g, x * psi), dim=1)


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


class ConvBlock(nn.Module):
    """ BN -> ReLU -> Conv3D """

    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.conv(x)


class ResDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            ResDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=False):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)
        self.conv = ConvBlock(in_channels // 2, out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class AttVNet(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels=8):
        super(AttVNet, self).__init__()
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

        self.up1 = UpConv(16 * n_channels, 8 * n_channels)
        self.att1 = AttnBlock(8 * n_channels, 8 * n_channels, 4 * n_channels)
        self.up_conv1 = ResDoubleConv(16 * n_channels, 8 * n_channels)

        self.up2 = UpConv(8 * n_channels, 4 * n_channels)
        self.att2 = AttnBlock(4 * n_channels, 4 * n_channels, 2 * n_channels)
        self.up_conv2 = ResDoubleConv(8 * n_channels, 4 * n_channels)

        self.up3 = UpConv(4 * n_channels, 2 * n_channels)
        self.att3 = AttnBlock(2 * n_channels, 2 * n_channels, n_channels)
        self.up_conv3 = ResDoubleConv(4 * n_channels, 2 * n_channels)

        self.up4 = UpConv(2 * n_channels, n_channels)
        self.att4 = AttnBlock(n_channels, n_channels, n_channels // 2, num_groups=4)
        self.up_conv4 = ResDoubleConv(2 * n_channels, n_channels)

        self.out = nn.Conv3d(in_channels=n_channels, out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)
        # x1:n -> x2:2n
        x2 = self.enc1(x1)
        # x2:2n -> x3:4n
        x3 = self.enc2(x2)
        # x3:4n -> x4:8n
        x4 = self.enc3(x3)
        # x4:8n -> bridge:16n
        bridge = self.bridge(x4)

        d4 = self.up1(bridge)
        x4 = self.att1(g=d4, x=x4)
        x4 = self.up_conv1(x4)

        d3 = self.up2(x4)
        x3 = self.att2(g=d3, x=x3)
        x3 = self.up_conv2(x3)

        d2 = self.up3(x3)
        x2 = self.att3(g=d2, x=x2)
        x2 = self.up_conv3(x2)

        d1 = self.up4(x2)
        x1 = self.att4(g=d1, x=x1)
        x1 = self.up_conv4(x1)

        return self.out(x1)


def main():
    use_cuda = torch.cuda.is_available()
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = AttVNet(4, 3, n_channels=32).to(device)
    # batch, channel, x, y, z
    x = torch.rand(4, 4, 155, 96, 96)
    x = x.to(device)
    model.forward(x)
    print("everything ok!")


if __name__ == '__main__':
    main()
