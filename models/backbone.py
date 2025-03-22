import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class ConvBNMish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = Mish()

    def forward(self, x):
        return self.mish(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        reduced_channels = in_channels // 2
        
        self.conv1 = ConvBNMish(in_channels, reduced_channels, 1)
        self.conv2 = ConvBNMish(reduced_channels, in_channels, 3)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class CSPStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()
        self.downsample = ConvBNMish(in_channels, out_channels, 3, stride=2)
        
        self.split_conv0 = ConvBNMish(out_channels, out_channels // 2, 1)
        self.split_conv1 = ConvBNMish(out_channels, out_channels // 2, 1)
        
        self.blocks = nn.Sequential(
            *[ResidualBlock(out_channels // 2) for _ in range(num_blocks)]
        )
        
        self.transition = ConvBNMish(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample(x)
        
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        
        x1 = self.blocks(x1)
        
        x = torch.cat([x0, x1], dim=1)
        return self.transition(x)

class CSPDarknet53(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Initial convolution
        self.conv1 = ConvBNMish(in_channels, 32, 3)
        
        # CSP stages with increasing channels and different numbers of blocks
        self.stage1 = CSPStage(32, 64, 1)    # P1
        self.stage2 = CSPStage(64, 128, 2)   # P2
        self.stage3 = CSPStage(128, 256, 8)  # P3
        self.stage4 = CSPStage(256, 512, 8)  # P4
        self.stage5 = CSPStage(512, 1024, 4) # P5

    def forward(self, x):
        features = {}
        
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        features["dark3"] = self.stage3(x)    # P3 output
        features["dark4"] = self.stage4(features["dark3"])  # P4 output
        features["dark5"] = self.stage5(features["dark4"])  # P5 output
        
        return features

if __name__ == "__main__":
    # Quick test
    model = CSPDarknet53()
    x = torch.randn(1, 3, 608, 608)
    features = model(x)
    for k, v in features.items():
        print(f"{k}: {v.shape}")
