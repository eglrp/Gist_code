# full assembly of the sub-parts to form the complete net

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)  # 输入层， n_channels=3 输入图像为3通道
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

        # 右侧 每次up时，左侧feature map需要先裁剪，再与右侧深度上合并
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        # 最后一层的卷积核大小为1*1，将64通道的特征图转化为特定深度（分类数量，二分类为2）的结果
        self.outc = outconv(64, n_classes)  # 输出层，n_classes=1 二分类

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # 64x64  维度512
        x5 = self.down4(x4)  # 28x28 维度1024
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
