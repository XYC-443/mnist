import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 1. 加载和预处理 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor(),  # 将图像转换为张量
                                transforms.Normalize((0.5,), (0.5,))])  # 标准化

# 下载训练集和测试集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


# 2. 定义神经网络模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 第一层卷积层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 第二层卷积层
        self.fc1 = nn.Linear(7 * 7 * 64, 128)  # 全连接层1
        self.fc2 = nn.Linear(128, 10)  # 全连接层2，输出10个类别

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # 激活函数ReLU
        x = torch.max_pool2d(x, 2)  # 最大池化
        x = torch.relu(self.conv2(x))  # 激活函数ReLU
        x = torch.max_pool2d(x, 2)  # 最大池化
        x = x.view(-1, 7 * 7 * 64)  # 展平为一维张量
        x = torch.relu(self.fc1(x))  # 激活函数ReLU
        x = self.fc2(x)  # 最后一层输出
        return x


# 初始化模型
model = SimpleCNN()

# 3. 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# 4. 训练模型
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # 将输入和标签转移到 GPU 如果可用
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
            model.cuda()

        # 零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 打印统计信息
        running_loss += loss.item()
        if i % 100 == 99:  # 每100个批次打印一次
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

print("Finished Training")

# 5. 测试模型
model.eval()  # 设置为评估模式
correct = 0
total = 0
with torch.no_grad():  # 不计算梯度
    for inputs, labels in testloader:
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')

