import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 定义卷积神经网络结构
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        # 第一层卷积：输入1通道（灰度图），输出32通道，卷积核3x3
        # 用32个不同的卷积核进行了32次处理
        # kernel_size=3表示卷积核大小为3x3，padding=1表示在输入图像周围添加1像素的零填充，以保持输出尺寸不变
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 第二层卷积：输入32通道，输出64通道
        # 用2个不同的卷积核进行了64次处理
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 池化层：减小图像尺寸
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 最终输出10类（0-9）
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # print("初始 输出内容:\n", x.shape) # 打印输入数据的形状
        # 卷积 -> 激活 -> 池化
        x = self.pool(F.relu(self.conv1(x))) # 28x28 -> 14x14
        # print("第一次卷积池化输出内容:\n", x.shape) 
        x = self.pool(F.relu(self.conv2(x))) # 14x14 -> 7x7
        # print("第二次卷积池化输出内容:\n", x.shape) 
        # 展平数据 64 张7×7 的特征图
        x = x.view(-1, 64 * 7 * 7)
        # print("view 输出内容:\n", x.shape)
        # 全连接层 -> 激活 -> Dropout
        x = F.relu(self.fc1(x))
        # print("fc1 输出内容:\n", x.shape)
        x = self.dropout(x)
        # 这里输出为0-9的概率分布
        x = self.fc2(x)
        # 👇 加上这一句，就能打印形状 + 内容
        # print("fc2 最终输出 shape:", x.shape)
        return x

# 2. 数据准备 (使用 MNIST 手写数字数据集)
def prepare_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # 归一化
    ])
    
    train_dataset = datasets.MNIST(root='./train_cache', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./train_cache', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# 3. 训练函数
def train_model():
    # 硬件检查 (如果有你的 RX 7900 XT 且配置了 ROCm，可以使用 'cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")

    # 创建模型、优化器和损失函数
    model = DigitCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = prepare_data()

    # 简单训练 3 轮 (Epochs)
    model.train()
    for epoch in range(1, 4):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            
            # (1) 搬运数据
            data, target = data.to(device), target.to(device)
            # (2) 清空梯度
            optimizer.zero_grad()
            # (3) 前向传播
            output = model(data)
            # (4) 计算损失
            loss = criterion(output, target)
            # (5) 反向传播  只负责计算梯度（求导）
            loss.backward()
            # (6) 更新参数  只负责更新参数（梯度下降）
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 200 == 0:
                print(f"Epoch {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")
    
    print("训练完成！")
    return model

# 运行
if __name__ == "__main__":
    digit_model = train_model()
    # 保存模型
    torch.save(digit_model.state_dict(), "digit_cnn.pth")
    print("模型已保存为 digit_cnn.pth")