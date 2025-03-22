import torch
import torch.nn as nn
from .backbone import ConvBNMish

class SPP(nn.Module):
    """Spatial Pyramid Pooling layer used in YOLOv3-SPP and YOLOv4"""
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBNMish(in_channels, hidden_channels, 1)
        
        self.maxpools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
            for k in kernel_sizes
        ])
        
        # Conv after concat of maxpool outputs and conv1
        self.conv2 = ConvBNMish(hidden_channels * (len(kernel_sizes) + 1), out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        pooled_results = [x]
        pooled_results.extend([mp(x) for mp in self.maxpools])
        x = torch.cat(pooled_results, dim=1)
        return self.conv2(x)

class PANetBottom(nn.Module):
    """Bottom-up path in PANet"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = ConvBNMish(in_channels, in_channels//2, 1)
        self.conv2 = ConvBNMish(in_channels//2, in_channels, 3, stride=2)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class PANetTop(nn.Module):
    """Top-down path in PANet"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = ConvBNMish(in_channels, in_channels//2, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.upsample(self.conv(x))

class YOLONeck(nn.Module):
    def __init__(self, channels_list=(256, 512, 1024)):
        super().__init__()
        c3, c4, c5 = channels_list
        
        # SPP block on the deepest layer
        self.spp = SPP(c5, c5)
        
        # Top-down path (upsampling)
        self.pat5 = PANetTop(c5)
        self.cv4 = ConvBNMish(c4 + c5//2, c4, 1)
        
        self.pat4 = PANetTop(c4)
        self.cv3 = ConvBNMish(c3 + c4//2, c3, 1)
        
        # Bottom-up path
        self.pab3 = PANetBottom(c3)
        self.cv3_up = ConvBNMish(c3 + c4, c4, 1)
        
        self.pab4 = PANetBottom(c4)
        self.cv4_up = ConvBNMish(c4 + c5, c5, 1)
        
        # Output convolutions
        self.out_conv3 = nn.Sequential(
            ConvBNMish(c3, c3, 3),
            ConvBNMish(c3, c3, 3)
        )
        self.out_conv4 = nn.Sequential(
            ConvBNMish(c4, c4, 3),
            ConvBNMish(c4, c4, 3)
        )
        self.out_conv5 = nn.Sequential(
            ConvBNMish(c5, c5, 3),
            ConvBNMish(c5, c5, 3)
        )

    def forward(self, features):
        c3, c4, c5 = features["dark3"], features["dark4"], features["dark5"]
        
        # SPP on deepest layer
        p5 = self.spp(c5)
        
        # Top-down path
        p5_td = self.pat5(p5)
        p4 = self.cv4(torch.cat([p5_td, c4], dim=1))
        
        p4_td = self.pat4(p4)
        p3 = self.cv3(torch.cat([p4_td, c3], dim=1))
        
        # Bottom-up path
        p3_up = self.pab3(p3)
        p4 = self.cv3_up(torch.cat([p3_up, p4], dim=1))
        
        p4_up = self.pab4(p4)
        p5 = self.cv4_up(torch.cat([p4_up, p5], dim=1))
        
        # Final processing
        p3 = self.out_conv3(p3)
        p4 = self.out_conv4(p4)
        p5 = self.out_conv5(p5)
        
        return {
            "p3": p3,  # Detection for small objects
            "p4": p4,  # Detection for medium objects
            "p5": p5   # Detection for large objects
        }

if __name__ == "__main__":
    # Quick test
    import torch
    neck = YOLONeck()
    features = {
        "dark3": torch.randn(1, 256, 76, 76),
        "dark4": torch.randn(1, 512, 38, 38),
        "dark5": torch.randn(1, 1024, 19, 19)
    }
    outputs = neck(features)
    for k, v in outputs.items():
        print(f"{k}: {v.shape}")
