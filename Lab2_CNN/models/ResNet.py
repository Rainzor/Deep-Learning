import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, use_skip=True,
                 cardinality=1, base_width=64):
        """
        通用的 BasicBlock，可用于 ResNet 和 ResNeXt

        Parameters:
        - in_channels (int): 输入通道数
        - out_channels (int): 输出通道数（不包括扩展）
        - stride (int): 步幅
        - down_sample (bool): 是否使用跳跃连接
        - cardinality (int): 分组数，ResNet 为1
        - base_width (int): 基础宽度，ResNet 通常为64
        """
        super(BasicBlock, self).__init__()
        self.expansion = BasicBlock.expansion
        self.use_skip = use_skip
        D = int(math.floor(out_channels * (base_width / 64)) * cardinality)
        C = cardinality

        # 第一层 3x3 卷积
        self.conv1 = nn.Conv2d(in_channels, D, kernel_size=3, stride=stride, 
                               padding=1, bias=False, groups=C if C > 1 else 1)
        self.bn1 = nn.BatchNorm2d(D)
        self.relu = nn.ReLU(inplace=True)

        # 第二层 3x3 卷积
        self.conv2 = nn.Conv2d(D, out_channels * self.expansion, kernel_size=3, 
                               stride=1, padding=1, bias=False, groups=C if C > 1 else 1)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

        # 跳跃连接
        if use_skip and (in_channels != out_channels * self.expansion or stride != 1):
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * self.expansion, 
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * self.expansion)
                )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_skip:
            out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, use_skip=True,
                 cardinality=32, base_width=4):
        """
        通用的 Bottleneck 块，可用于 ResNet 和 ResNeXt

        Parameters:
        - in_channels (int): 输入通道数
        - out_channels (int): 输出通道数（不包括扩展）
        - stride (int): 步幅
        - down_sample (bool): 是否使用跳跃连接
        - cardinality (int): 分组数，ResNet 为1，ResNeXt 通常为32
        - base_width (int): 基础宽度，ResNet 通常为64，ResNeXt 通常为4
        """
        super(Bottleneck, self).__init__()
        self.expansion = Bottleneck.expansion
        self.use_skip = use_skip
        D = int(math.floor(out_channels * (base_width / 64)) * cardinality)
        C = cardinality

        # 第一层 1x1 卷积（减少通道数）
        self.conv1 = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, 
                               bias=False)
        self.bn1 = nn.BatchNorm2d(D)

        # 第二层 3x3 分组卷积
        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=stride, 
                               padding=1, bias=False, groups=C if C > 1 else 1)
        self.bn2 = nn.BatchNorm2d(D)

        # 第三层 1x1 卷积（恢复通道数）
        self.conv3 = nn.Conv2d(D, out_channels * self.expansion, kernel_size=1, 
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        # 跳跃连接
        if  use_skip and (in_channels != out_channels * self.expansion or stride != 1):
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * self.expansion, 
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * self.expansion)
                )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.use_skip:
            out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, config, output_dim, use_skip=True):
        """
        通用的 ResNet/ResNeXt 网络结构

        Parameters:
        - config (tuple): (块类型, 每层的块数, 每层的输出通道数)
        - output_dim (int): 输出维度，如分类任务的类别数
        - block (nn.Module): 基本块类型，UnifiedBasicBlock 或 UnifiedBottleneck
        - cardinality (int): 分组数，ResNet 为1，ResNeXt 通常为32
        - base_width (int): 基础宽度，ResNet 为64，ResNeXt 通常为4
        """
        super(ResNet, self).__init__()

        block, n_blocks, channels, cardinality, base_width = config
        self.in_channels = channels[0]
        self.cardinality = cardinality
        self.base_width = base_width
        self.use_skip = use_skip
        assert len(n_blocks) == len(channels) == 4

        # 与原始 ResNet 不同，我们使用 kernel_size=3, stride=1 的第一个卷积层
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Identity()

        self.res_layers = nn.ModuleList()
        for i in range(4):
            stride = 1 if i == 0 else 2
            self.res_layers.append(self.get_resnet_layer(block, n_blocks[i],
                                                            channels[i], stride))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, output_dim)

    def get_resnet_layer(self, block, n_blocks, channels, stride=1):
        layers = []

        layers.append(block(self.in_channels, channels, stride, use_skip=self.use_skip,
                           cardinality=self.cardinality, base_width=self.base_width))

        for _ in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels, stride=1, use_skip=self.use_skip,
                               cardinality=self.cardinality, base_width=self.base_width))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.res_layers:
            x = layer(x)

        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)

        return x, h

