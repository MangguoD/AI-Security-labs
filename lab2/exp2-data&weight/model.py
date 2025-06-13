# model.py
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7*7*64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B,32,14,14]
        x = self.pool(F.relu(self.conv2(x)))  # [B,64,7,7]
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 测试网络输出形状
if __name__ == '__main__':
    import torch
    net = LeNet()
    x = torch.randn(8,1,28,28)
    y = net(x)
    print(y.shape)  # 应为 [8,10]