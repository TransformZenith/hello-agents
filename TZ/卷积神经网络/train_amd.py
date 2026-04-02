import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch_directml  # 1. 导入 DirectML 库

# 1. 定义卷积神经网络结构 (代码保持不变)
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 2. 数据准备 (保持不变)
def prepare_data(batch_size=1024): # 1. 把 Batch 提高到 1024 甚至更高
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./train_cache', train=True, download=True, transform=transform)
    # 2. 开启多进程和内存锁定
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,       # 使用 4 个 CPU 核心同时读图，防止 GPU 等数据
        pin_memory=True      # 锁定内存页，让数据从内存传到显存的速度翻倍
    )
    
    return train_loader
# 3. 训练函数 (重点修改这里)
def train_model():
    # --- 修改点：使用 DirectML 设备 ---
    device = torch_directml.device() 
    print(f"检测到 AMD 显卡，当前使用设备: {device}")

    # 创建模型并搬运到 AMD 显存
    model = DigitCNN().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_loader = prepare_data()

    model.train()
    for epoch in range(1, 4):
        for batch_idx, (data, target) in enumerate(train_loader):
            # 将数据也搬运到相同设备
            data, target = data.to(device), target.to(device)
            if batch_idx == 0:
                print(f"数据当前所在设备: {data.device}") # 应该显示 privateuseone:0
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # 建议修改为：
            if batch_idx % 20 == 0:
                # 进度百分比计算：使用当前已处理的图片数 / 总数
                processed = batch_idx * len(data)
                total = len(train_loader.dataset)
                print(f"Epoch {epoch} [{processed}/{total} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.4f}")

    print("训练完成！")
    # 训练完后，建议把模型搬回 CPU 再返回/保存，这样最稳妥
    return model.to("cpu") 

if __name__ == "__main__":
    digit_model = train_model()
    # 保存权重
    torch.save(digit_model.state_dict(), "digit_cnn.pth")
    print("模型已保存为 digit_cnn.pth")