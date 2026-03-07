# Task 04: Common CNN Architectures - Tiny-ImageNet分类
# 任务：使用不同策略训练VGG16模型

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 数据加载函数（简化版本，使用CIFAR-10代替Tiny-ImageNet进行快速测试）
def load_data():
    """加载数据集（使用CIFAR-10作为演示）"""
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 划分训练集和验证集
    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

# 训练函数
def train_model(model, train_loader, val_loader, epochs=5, lr=0.001):
    """训练模型并记录历史"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_acc': [], 'val_acc': [], 'epoch_time': []}

    for epoch in range(epochs):
        start_time = time.time()

        # 训练阶段
        model.train()
        correct, total = 0, 0
        for inputs, labels in train_loader:
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

        # 验证阶段
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        epoch_time = time.time() - start_time

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)

        print(f'Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, Time={epoch_time:.2f}s')

    return history

# 测试函数
def test_model(model, test_loader):
    """测试模型准确率和推理时间"""
    model.eval()
    correct, total = 0, 0
    inference_times = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            start_time = time.time()
            outputs = model(inputs)
            inference_times.append((time.time() - start_time) / inputs.size(0))

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    avg_inference_time = np.mean(inference_times) * 1000  # 转换为毫秒

    return test_acc, avg_inference_time

print("数据加载中...")
train_loader, val_loader, test_loader = load_data()
print("数据加载完成！")

# 实验1: VGG16从头训练
print("\n=== 实验1: VGG16从头训练 ===")
model1 = models.vgg16(weights=None)
model1.classifier[6] = nn.Linear(4096, 10)
model1 = model1.to(DEVICE)
history1 = train_model(model1, train_loader, val_loader, epochs=5)
test_acc1, inf_time1 = test_model(model1, test_loader)
print(f"测试准确率: {test_acc1:.2f}%, 推理时间: {inf_time1:.4f}ms/图像")

# 实验2: VGG16迁移学习（仅训练分类层）
print("\n=== 实验2: VGG16迁移学习（冻结特征层）===")
model2 = models.vgg16(weights='IMAGENET1K_V1')
for param in model2.features.parameters():
    param.requires_grad = False
model2.classifier[6] = nn.Linear(4096, 10)
model2 = model2.to(DEVICE)
history2 = train_model(model2, train_loader, val_loader, epochs=5)
test_acc2, inf_time2 = test_model(model2, test_loader)
print(f"测试准确率: {test_acc2:.2f}%, 推理时间: {inf_time2:.4f}ms/图像")

# 实验3: VGG16微调（训练整个网络）
print("\n=== 实验3: VGG16微调（训练全部层）===")
model3 = models.vgg16(weights='IMAGENET1K_V1')
model3.classifier[6] = nn.Linear(4096, 10)
model3 = model3.to(DEVICE)
history3 = train_model(model3, train_loader, val_loader, epochs=5, lr=0.0001)
test_acc3, inf_time3 = test_model(model3, test_loader)
print(f"测试准确率: {test_acc3:.2f}%, 推理时间: {inf_time3:.4f}ms/图像")

# 实验4: ResNet18（自选模型）
print("\n=== 实验4: ResNet18（自选模型）===")
model4 = models.resnet18(weights='IMAGENET1K_V1')
model4.fc = nn.Linear(512, 10)
model4 = model4.to(DEVICE)
history4 = train_model(model4, train_loader, val_loader, epochs=5)
test_acc4, inf_time4 = test_model(model4, test_loader)
print(f"测试准确率: {test_acc4:.2f}%, 推理时间: {inf_time4:.4f}ms/图像")

# Task 4.1: 绘制训练和验证准确率曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history1['train_acc'], label='VGG16从头训练')
plt.plot(history2['train_acc'], label='VGG16迁移学习')
plt.plot(history3['train_acc'], label='VGG16微调')
plt.plot(history4['train_acc'], label='ResNet18')
plt.xlabel('训练轮次')
plt.ylabel('训练准确率 (%)')
plt.title('训练准确率对比')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history1['val_acc'], label='VGG16从头训练')
plt.plot(history2['val_acc'], label='VGG16迁移学习')
plt.plot(history3['val_acc'], label='VGG16微调')
plt.plot(history4['val_acc'], label='ResNet18')
plt.xlabel('训练轮次')
plt.ylabel('验证准确率 (%)')
plt.title('验证准确率对比')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('Task4_1_accuracy_curves.png', dpi=300, bbox_inches='tight')
print("\n图表已保存: Task4_1_accuracy_curves.png")

# Task 4.1: 创建性能对比表格
results_df = pd.DataFrame({
    '模型': ['VGG16从头训练', 'VGG16迁移学习', 'VGG16微调', 'ResNet18'],
    '测试准确率 (%)': [test_acc1, test_acc2, test_acc3, test_acc4],
    '每轮训练时间 (s)': [np.mean(history1['epoch_time']), np.mean(history2['epoch_time']),
                      np.mean(history3['epoch_time']), np.mean(history4['epoch_time'])],
    '推理时间 (ms/图像)': [inf_time1, inf_time2, inf_time3, inf_time4]
})

print("\n=== Task 4.1: 性能对比表格 ===")
print(results_df.to_string(index=False))
results_df.to_csv('Task4_1_results_table.csv', index=False, encoding='utf-8-sig')
print("\n表格已保存: Task4_1_results_table.csv")

# Task 4.2: 最佳模型测试结果可视化
best_idx = results_df['测试准确率 (%)'].idxmax()
best_model_name = results_df.loc[best_idx, '模型']
best_acc = results_df.loc[best_idx, '测试准确率 (%)']

plt.figure(figsize=(8, 6))
bars = plt.bar(results_df['模型'], results_df['测试准确率 (%)'],
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
bars[best_idx].set_color('#FFD700')
plt.xlabel('模型')
plt.ylabel('测试准确率 (%)')
plt.title(f'不同模型在测试集上的准确率对比\n最佳模型: {best_model_name} ({best_acc:.2f}%)')
plt.xticks(rotation=15, ha='right')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('Task4_2_test_accuracy.png', dpi=300, bbox_inches='tight')
print("\n图表已保存: Task4_2_test_accuracy.png")

print(f"\n使用的GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("\n=== Task 04 完成 ===")
