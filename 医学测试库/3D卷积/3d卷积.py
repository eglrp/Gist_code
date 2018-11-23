import torch
import torch.nn as nn


m = nn.Conv3d(2, 1, 3, stride=2)
input = torch.randn(1, 2, 3, 5, 6)
output = m(input)

print("卷积的权重：")
print(m.weight)
print("卷积的偏重：")
print(m.bias)

print("二维卷积后的输出：")
print(output)
print("输出的尺度：")
print(output.size())

#测试