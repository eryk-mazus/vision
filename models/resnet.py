import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Shortcut connection with 1x1 conv for adjusting
# the number of channels, and stride to adjust the size
class ShortcutProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.bn(self.conv(x))

class ResidualLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super(ResidualLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.downsample = ShortcutProjection(in_channels, out_channels, stride)

    def forward(self, x):
        shortcut = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + shortcut)

class BottleneckLayer(nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int, stride: int):
        super(BottleneckLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.downsample = ShortcutProjection(in_channels, out_channels, stride)
    
    def forward(self, x):
        shortcut = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + shortcut)

class ResNet(nn.Module):
    def __init__(self, input_img_channels):
        super(ResNet, self).__init__()
        # TODO: flexible design
        # TODO: to replace 64 down here:
        self.conv = nn.Conv2d(input_img_channels, 64, kernel_size=7, padding=3, stride=2)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        ...
    

if __name__ == "__main__":
    ...
    # kaiming
    # conv3d biases
