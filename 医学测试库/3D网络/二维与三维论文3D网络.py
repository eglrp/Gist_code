import torch.nn as nn
import torch as t

class Net_3D(nn.Module):
    '''
        定义ResNet18网络
        '''

    def __init__(self):
        super(Net_3D, self).__init__()

        # 特征层
        self.features = nn.Sequential(
            t.nn.Conv3d(in_channels=1,out_channels=64,kernel_size=(5,5,3)),
            t.nn.ReLU(),
            t.nn.MaxPool3d(kernel_size=(2,2,2)),
            t.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(5, 5, 3)),
            t.nn.ReLU(),
            t.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(5, 5, 3)),
            t.nn.ReLU()
        )

        # 分类层
        self.classifier = nn.Sequential(
            t.nn.Linear(28800, 250),
            # t.nn.Dropout(p=0.5),
            t.nn.ReLU(),
            # t.nn.Dropout(p=0.5),
            t.nn.Linear(250, 2)
        )

        # 参数初始化

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # 是否接softmax
        return x



if __name__ == '__main__':
    # 归一化
    model=Net_3D()
    input = t.randn(1, 1, 26, 40, 40)  #1 batch size    1  输入通道     26，40,40为一个通道的样本
    output=model(input)
    print(output.size())