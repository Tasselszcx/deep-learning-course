# Task 09: 扩散模型
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Task 9.1: 实现add_noise函数
def add_noise(x, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """为图像添加噪声"""
    noise = torch.randn_like(x)
    sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    x_noisy = sqrt_alpha_t * x + sqrt_one_minus_alpha_t * noise
    return x_noisy, noise

# 设置扩散参数
timesteps = 1000
betas = torch.linspace(0.0001, 0.02, timesteps)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

# 加载MNIST
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 获取一张图像
img, _ = next(iter(loader))
img = img.to(DEVICE)

# Task 9.1: 生成不同时间步的加噪图像
t_steps = [0, 250, 500, 750, 999]
fig, axes = plt.subplots(3, len(t_steps), figsize=(15, 6))

for i, t in enumerate(t_steps):
    t_tensor = torch.tensor([t]).to(DEVICE)
    noisy_img, noise = add_noise(img, t_tensor,
                                 sqrt_alphas_cumprod.to(DEVICE),
                                 sqrt_one_minus_alphas_cumprod.to(DEVICE))

    axes[0, i].imshow(img[0, 0].cpu(), cmap='gray')
    axes[0, i].set_title(f't={t}')
    axes[0, i].axis('off')

    axes[1, i].imshow(noisy_img[0, 0].cpu(), cmap='gray')
    axes[1, i].axis('off')

    axes[2, i].imshow(noise[0, 0].cpu(), cmap='gray')
    axes[2, i].axis('off')

axes[0, 0].set_ylabel('原始图像', fontsize=12)
axes[1, 0].set_ylabel('加噪图像', fontsize=12)
axes[2, 0].set_ylabel('噪声', fontsize=12)
plt.tight_layout()
plt.savefig('Task9_1_noise_process.png', dpi=300, bbox_inches='tight')
print("Task 9.1 图表已保存: Task9_1_noise_process.png")

# Task 9.2: 简化的扩散模型训练
class SimpleDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x, t):
        return self.net(x)

model = SimpleDiffusionModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 训练
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
epochs = 5
losses = []

for epoch in range(epochs):
    epoch_loss = 0
    for batch_img, _ in train_loader:
        batch_img = batch_img.to(DEVICE)
        t = torch.randint(0, timesteps, (batch_img.size(0),)).to(DEVICE)
        noisy_img, noise = add_noise(batch_img, t,
                                     sqrt_alphas_cumprod.to(DEVICE),
                                     sqrt_one_minus_alphas_cumprod.to(DEVICE))

        optimizer.zero_grad()
        pred_noise = model(noisy_img, t)
        loss = criterion(pred_noise, noise)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

# Task 9.2: 绘制训练损失
plt.figure(figsize=(10, 6))
plt.plot(losses, marker='o')
plt.xlabel('训练轮次')
plt.ylabel('损失')
plt.title('扩散模型训练损失曲线')
plt.grid(True)
plt.savefig('Task9_2_training_loss.png', dpi=300, bbox_inches='tight')
print("Task 9.2 图表已保存: Task9_2_training_loss.png")
print("\n=== Task 09 完成 ===")
