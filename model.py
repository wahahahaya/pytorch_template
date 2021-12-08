from re import U
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_channel: int, out_channel: int, stride: int=1):
    return nn.Conv2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=1,
        stride=stride,
        bias=False
    )

def conv3x3(in_channel: int, out_channel: int, stride: int=1, dilation: int=1):
    return nn.Conv2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        bias=False,
        dilation=dilation
    )

class Bottleneck(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, mid_channel: int=None , stride: int=1, dilation: int=1):
        super(Bottleneck, self).__init__()

        if not mid_channel:
            mid_channel=out_channel
        self.conv1 = conv1x1(in_channel, mid_channel)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.conv2 = conv3x3(mid_channel, mid_channel, stride, dilation)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.conv3 = conv1x1(mid_channel, out_channel)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.downsample = nn.Sequential(
            conv1x1(in_channel, out_channel),
            nn.BatchNorm2d(out_channel),
        )
        self.in_channel = in_channel

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x.clone())

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        
        x += identity
        out = self.relu(x)

        return(out)

class Down(nn.Module):
    """Downscaling with maxpool then bottleneck"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Bottleneck(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then bottleneck"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = Bottleneck(in_channel=in_channels, out_channel=out_channels, mid_channel=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = Bottleneck(in_channel=in_channels, out_channel=out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ResUNet(nn.Module):
    def __init__(self, out_channel, bilinear=True):
        super(ResUNet, self).__init__()

        self.inc = Bottleneck(in_channel=1, out_channel=32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64// factor, bilinear)
        self.up4 = Up(64, 32, bilinear)

        self.outc = nn.Conv2d(32, out_channel, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1,1,196,512).to(device)
    model = ResUNet(out_channel=1).to(device)
    print(model(x).shape)