# Task 04 简化版 - 只用ResNet18快速测试
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 加载CIFAR-10
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_size = int(0.9 * len(trainset))
val_size = len(trainset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

print("数据加载完成")

# 简单训练函数
def train_quick(model, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    history = {'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for i, (inputs, labels) in enumerate(train_loader):
            if i > 50:  # 只训练50个batch快速测试
                break
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                if i > 20:  # 只验证20个batch
                    break
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        print(f'Epoch {epoch+1}/{epochs}: Train={train_acc:.1f}%, Val={val_acc:.1f}%')

    return history

# 只训练ResNet18
print("\n训练ResNet18...")
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(512, 10)
model = model.to(DEVICE)
history = train_quick(model, epochs=3)

# 简单绘图
plt.figure(figsize=(8, 5))
plt.plot(history['train_acc'], label='训练准确率')
plt.plot(history['val_acc'], label='验证准确率')
plt.xlabel('轮次')
plt.ylabel('准确率 (%)')
plt.title('ResNet18训练结果')
plt.legend()
plt.grid(True)
plt.savefig('Task4_quick_test.png', dpi=300, bbox_inches='tight')
print("\n图表已保存: Task4_quick_test.png")
print("快速测试完成！")
