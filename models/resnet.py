import torch
from torch import nn
<<<<<<< HEAD
from typing import List, Optional
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
=======
from typing import List
>>>>>>> a9e6ba98231577e597446a477041dacb73bdd19e

# Shortcut connection with 1x1 conv for adjusting
# the number of channels, and stride to adjust the size
class ShortcutProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.bn(self.conv(x))

class ResidualLayer(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super(ResidualLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.downsample = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        shortcut = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + shortcut)

class BottleneckLayer(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, bottleneck_channels: int, stride: int):
        super(BottleneckLayer, self).__init__()
        self.out_channels = bottleneck_channels*self.expansion
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.conv3 = nn.Conv2d(bottleneck_channels, self.out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != self.out_channels:
            self.downsample = ShortcutProjection(in_channels, self.out_channels, stride)
        else:
            self.downsample = nn.Identity()
    
    def forward(self, x):
        shortcut = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + shortcut)

class ResNet(nn.Module):
    def __init__(self, block: nn.Module, input_img_channels: int, n_blocks: List[int], n_channels: List[int], num_classes: int):
        super(ResNet, self).__init__()
        if len(n_blocks) != len(n_channels):
            raise ValueError('length of `n_blocks` must be equal to length of `n_channels`')
        
        self.conv = nn.Conv2d(input_img_channels, n_channels[0], kernel_size=7, padding=3, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        blocks = []
        prev_channels = n_channels[0]
        for i, number_of_blocks in enumerate(n_blocks):
            stride = 2 if i != 0 else 1
            blocks.append(block(prev_channels, n_channels[i], stride))
            prev_channels = n_channels[i] * block.expansion
            for _ in range(1, number_of_blocks):
                blocks.append(block(prev_channels, n_channels[i], 1))

        self.blocks = nn.Sequential(*blocks)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_channels[-1]*block.expansion, num_classes)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.maxpool(out)
        out = self.blocks(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def resnet34(input_channels, num_classes):
    return ResNet(ResidualLayer, input_channels, [3,4,6,3], [64,128,256,512], num_classes)

def resnet50(input_channels, num_classes):
    return ResNet(BottleneckLayer, input_channels, [3,4,6,3], [64,128,256,512], num_classes)


if __name__ == "__main__":
    # tests
    BATCH_SIZE = 2
    IMG_CHANNELS = 3
    example = torch.randn(BATCH_SIZE, IMG_CHANNELS, 224, 224)
    m = ResNet(block=ResidualLayer, input_img_channels=IMG_CHANNELS, n_blocks=[3,4,6,3], n_channels=[64,128,256,512], num_classes=10)
    # m = ResNet(block=BottleneckLayer, input_img_channels=IMG_CHANNELS, n_blocks=[3,4,6,3], n_channels=[64,128,256,512], num_classes=10)
    output = m(example)

    print('done')
