import torch
import torch.nn as nn
from utilities import FFres_block

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        # 初始化普通卷积层
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=3, stride=1, padding=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        return x


class F2ResNet(nn.Module):

    def __init__(self, num_classes=1000, in_channels=64,
                 groups=1, width_per_group=64):
        super(F2ResNet, self).__init__()



        # TODO add ratio-inplanes-groups assertion

        self.in_channels = in_channels
        self.stride = 1
        self.groups = groups
        self.base_width = width_per_group

        # 定义第一个卷积层
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        # 定义第一个批标准化层
        self.bn1 = nn.BatchNorm2d(in_channels)

        # 定义ReLU激活函数
        self.relu = nn.ReLU(inplace=True)

        # 定义最大池化层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 定义自适应平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        #全连接层
        self.fc = nn.Linear(in_channels * 8, num_classes)

    def res_layer(self, x, in_channels, out_channels, stride=1):
        # 定义基本残差块
        x_l = BasicBlock(x, in_channels, out_channels)
        # 定义快速傅里叶残差块
        x_g = FFres_block(x, in_channels, out_channels)
        # 定义卷积层
        x = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        # 返回残差块相加的输出
        return x + x_l + x_g


    def forward(self, x):

        # 将输入张量传递给第一个卷积层
        x = self.conv1(x)
        # 将卷积层的输出传递给归一化层
        x = self.bn1(x)
        # 对归一化层的输出应用 ReLU 激活函数
        x = self.relu(x)
        # 对激活后的输出执行最大池化操作
        x = self.maxpool(x)

        # 将经过最大池化的输出传递给模型的第一个 ResNet 层
        x = self.res_layer(self.in_channels * 1, self.in_channels * 2, stride=1)
        # 将第一层的输出传递给模型的第二个 ResNet 层
        x = self.res_layer(self.in_channels * 2, self.in_channels * 4, stride=1)
        # 将第二层的输出传递给模型的第三个 ResNet 层
        x = self.res_layer(self.in_channels * 4, self.in_channels * 8, stride=1)
        # 将第三层的输出传递给模型的第四个 ResNet 层
        x = self.res_layer(self.in_channels * 8, self.in_channels * 8, stride=1)
        # 对第四层输出的局部特征进行自适应平均池化操作

        x = self.avgpool(x[0])
        # 重塑张量，使其具有与输入相同的批量大小，并保留所有其他维
        x = x.view(x.size(0), -1)
        # 将重塑后的张量传递给全连接层，以获取模型的最终输出
        x = self.fc(x)

        return x

model = F2ResNet()
torch.save(model, 'F2ResNet.h5')
