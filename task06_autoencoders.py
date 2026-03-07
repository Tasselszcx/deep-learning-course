# Task 06: Autoencoders - 表示学习
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载MNIST数据
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 非卷积自编码器
class MLPAutoencoder(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# 卷积自编码器
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*7*7, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32*7*7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

def train_autoencoder(model, train_loader, epochs=5):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for data, _ in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            recon, _ = model(data)
            loss = criterion(recon.view(-1, 784), data.view(-1, 784))
            loss.backward()
            optimizer.step()
    return model

def extract_features(model, loader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for data, label in loader:
            data = data.to(DEVICE)
            _, z = model(data)
            features.append(z.cpu().numpy())
            labels.append(label.numpy())
    return np.vstack(features), np.hstack(labels)

print("训练非卷积自编码器...")
mlp_ae = train_autoencoder(MLPAutoencoder(), train_loader)
print("训练卷积自编码器...")
conv_ae = train_autoencoder(ConvAutoencoder(), train_loader)

# 提取特征
print("提取特征...")
X_train = train_dataset.data.numpy().reshape(-1, 784) / 255.0
y_train = train_dataset.targets.numpy()
X_test = test_dataset.data.numpy().reshape(-1, 784) / 255.0
y_test = test_dataset.targets.numpy()

# PCA
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 自编码器特征
X_train_mlp, _ = extract_features(mlp_ae, train_loader)
X_test_mlp, _ = extract_features(mlp_ae, test_loader)
X_train_conv, _ = extract_features(conv_ae, train_loader)
X_test_conv, _ = extract_features(conv_ae, test_loader)

# Task 6.1: 训练线性分类器并对比准确率
results = {}
for name, X_tr, X_te in [('PCA', X_train_pca, X_test_pca),
                          ('非卷积AE', X_train_mlp, X_test_mlp),
                          ('卷积AE', X_train_conv, X_test_conv)]:
    clf = LogisticRegression(max_iter=100)
    clf.fit(X_tr, y_train)
    acc = clf.score(X_te, y_test) * 100
    results[name] = acc
    print(f"{name}: {acc:.2f}%")

df = pd.DataFrame(list(results.items()), columns=['方法', '测试准确率(%)'])
print("\n=== Task 6.1: 分类准确率对比 ===")
print(df.to_string(index=False))
df.to_csv('Task6_1_results.csv', index=False, encoding='utf-8-sig')

# Task 6.2: 重构质量对比
test_sample = test_dataset.data[:10].float().unsqueeze(1) / 255.0
test_sample = test_sample.to(DEVICE)

mlp_recon, _ = mlp_ae(test_sample)
conv_recon, _ = conv_ae(test_sample)

mse_mlp = nn.MSELoss()(mlp_recon.view(-1, 784), test_sample.view(-1, 784)).item()
mse_conv = nn.MSELoss()(conv_recon.view(-1, 784), test_sample.view(-1, 784)).item()

recon_df = pd.DataFrame({'方法': ['非卷积AE', '卷积AE'], 'MSE': [mse_mlp, mse_conv]})
print("\n=== Task 6.2: 重构质量对比 ===")
print(recon_df.to_string(index=False))
recon_df.to_csv('Task6_2_reconstruction.csv', index=False, encoding='utf-8-sig')

# 可视化
fig, axes = plt.subplots(3, 5, figsize=(10, 6))
for i in range(5):
    axes[0, i].imshow(test_sample[i].cpu().squeeze(), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(mlp_recon[i].cpu().detach().view(28, 28), cmap='gray')
    axes[1, i].axis('off')
    axes[2, i].imshow(conv_recon[i].cpu().detach().view(28, 28), cmap='gray')
    axes[2, i].axis('off')
axes[0, 0].set_ylabel('原始', fontsize=12)
axes[1, 0].set_ylabel('非卷积AE', fontsize=12)
axes[2, 0].set_ylabel('卷积AE', fontsize=12)
plt.tight_layout()
plt.savefig('Task6_2_reconstruction_visual.png', dpi=300, bbox_inches='tight')
print("图表已保存: Task6_2_reconstruction_visual.png")
print("\n=== Task 06 完成 ===")
