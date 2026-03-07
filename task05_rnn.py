# Task 05: RNN时间序列预测
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 简单RNN模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def train_rnn(X_train, y_train, X_test, y_test, epochs=50):
    model = SimpleRNN().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train_t = torch.FloatTensor(X_train).unsqueeze(-1).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).unsqueeze(-1).to(DEVICE)
        predictions = model(X_test_t).cpu().numpy()

    return predictions

# 加载数据
print("下载数据...")
import urllib.request
urllib.request.urlretrieve('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv',
                          'airline-passengers.csv')

data = pd.read_csv("airline-passengers.csv", usecols=[1], engine="python")
data_np = data.to_numpy(dtype="float32")

split_idx = int(len(data_np) * 0.7)
train_np, test_np = data_np[:split_idx], data_np[split_idx:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_norm = scaler.fit_transform(train_np)
test_norm = scaler.transform(test_np)

# Task 5.1 & 5.2: 不同窗口大小的实验
window_sizes = [3, 6, 12]
results = {}

for ws in window_sizes:
    print(f"\n训练窗口大小={ws}的模型...")
    X_train, y_train = create_sequences(train_norm.flatten(), ws)
    X_test, y_test = create_sequences(test_norm.flatten(), ws)

    predictions = train_rnn(X_train, y_train, X_test, y_test)
    predictions_rescaled = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = np.mean((predictions_rescaled - actual) ** 2)
    results[ws] = {'pred': predictions_rescaled, 'actual': actual, 'mse': mse}
    print(f"窗口大小={ws}, MSE={mse:.2f}")

# Task 5.1: 绘制预测曲线
plt.figure(figsize=(12, 6))
for ws in window_sizes:
    plt.plot(results[ws]['pred'], label=f'窗口大小={ws}')
plt.plot(results[window_sizes[0]]['actual'], 'k--', label='实际值', alpha=0.5)
plt.xlabel('时间步')
plt.ylabel('乘客数量')
plt.title('不同窗口大小的RNN预测结果对比')
plt.legend()
plt.grid(True)
plt.savefig('Task5_1_predictions.png', dpi=300, bbox_inches='tight')
print("\n图表已保存: Task5_1_predictions.png")

# Task 5.2: 性能对比表格
perf_df = pd.DataFrame({
    '窗口大小': window_sizes,
    'MSE': [results[ws]['mse'] for ws in window_sizes]
})
print("\n=== Task 5.2: 性能对比表格 ===")
print(perf_df.to_string(index=False))
perf_df.to_csv('Task5_2_results.csv', index=False, encoding='utf-8-sig')

# Task 5.3: 最佳模型的预测vs实际
best_ws = min(window_sizes, key=lambda x: results[x]['mse'])
plt.figure(figsize=(10, 6))
plt.plot(results[best_ws]['actual'], 'b-', label='实际值', linewidth=2)
plt.plot(results[best_ws]['pred'], 'r--', label=f'预测值(窗口={best_ws})', linewidth=2)
plt.xlabel('时间步')
plt.ylabel('乘客数量')
plt.title(f'最佳模型预测结果 (窗口大小={best_ws}, MSE={results[best_ws]["mse"]:.2f})')
plt.legend()
plt.grid(True)
plt.savefig('Task5_3_best_model.png', dpi=300, bbox_inches='tight')
print("图表已保存: Task5_3_best_model.png")
print("\n=== Task 05 完成 ===")
