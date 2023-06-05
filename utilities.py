import torch
import torch.nn as nn


class FastFourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FastFourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        # 批标准化层
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        # ReLU激活函数
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        # (batch, c, h, w//2+1, 2)
        # 对输入特征图进行2D快速傅里叶变换，并获取实部和虚部
        ffted = torch.fft.fft2(x)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        # (batch, c, 2, h, w//2+1)
        # 调整张量形状以适应卷积层的输入
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        # 将处理后的张量输入到卷积层、批标准化层和ReLU激活函数中
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w//2+1)
        ffted = self.relu(self.bn(ffted))

        # 调整输出张量的形状，以便在最后一个维度上表示复数（实部和虚部）
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch, c, h, w//2+1, 2)

        # 对处理后的复数张量执行2D逆傅里叶变换，以将其从频域返回到时域
        ffted_complex = torch.view_as_complex(ffted)
        output = torch.fft.ifft2(ffted_complex)
        # 返回逆傅里叶变换的实部作为输出
        return output.real


class FFres_block(nn.Module): #快速傅里叶残差块

    def __init__(self, in_channels, out_channels, groups=1):
        super(FFres_block, self).__init__()
        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.relu1 = torch.nn.ReLU(inplace=True)
        # 初始化快速傅立叶卷积层
        self.ffu = FastFourierUnit(in_channels, out_channels, groups=groups)
        # 定义第二个批标准化层
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        # 定义第二个ReLU激活函数
        self.relu2 = torch.nn.ReLU(inplace=True)
        # 定义卷积层
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=3, stride=1, padding=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.ffu(x)
        x = self.bn2(x)
        x = self.relu2(x)
        #3*3卷积
        x = self.conv(x)
        return x