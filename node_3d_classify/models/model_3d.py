import torch.nn as nn
import torch as t

class Net_3D(nn.Module):
    '''
    天池冠军的3D分类网络
    '''

    def __init__(self):
        super(Net_3D, self).__init__()

        # 特征层
        self.features = nn.Sequential(  # 未加激活和BN
            t.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 3)),
            t.nn.ReLU(),
            t.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3)),
            t.nn.ReLU(),
            t.nn.MaxPool3d(kernel_size=(1, 2, 2)),  # kernel_size =(kD,kH,kW)
            t.nn.Dropout(0.5),
            t.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3)),
            t.nn.ReLU(),
            t.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3)),
            t.nn.ReLU(),
            t.nn.MaxPool3d(kernel_size=(2, 2, 2)),
            t.nn.Dropout(0.5),
            t.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3)),
            t.nn.ReLU(),
            t.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3)),
            t.nn.ReLU(),
            t.nn.MaxPool3d(kernel_size=(2, 2, 2)),
            t.nn.Dropout(0.5),
        )

        # 分类层
        self.classifier = nn.Sequential(
            t.nn.Linear(128, 512), #  这儿有问题？
            t.nn.Dropout(0.5),
            t.nn.Linear(512, 512),
            t.nn.Dropout(0.5),
            t.nn.Linear(512, 2)
        )

        # 参数初始化

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # 是否接softmax
        return x



if __name__ == '__main__':
    model=Net_3D()
    input = t.randn(1, 1, 20, 36, 36)  #1 batch size    1  输入通道     26，40,40为一个通道的样本
    output=model(input)
    print(output.size())