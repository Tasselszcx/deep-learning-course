# Task 08: 强化学习 - Q-learning vs SARSA
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections
import random
import gymnasium as gym

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def softmax(x, temperature=0.025):
    x = (x - np.expand_dims(np.max(x, 1), 1)) / temperature
    e_x = np.exp(x)
    return e_x / (np.expand_dims(e_x.sum(1), -1) + 1e-5)

class DQNAgent:
    def __init__(self, state_size, action_size, policy='epsilon', algorithm='qlearning'):
        self.state_size = state_size
        self.action_size = action_size
        self.policy = policy
        self.algorithm = algorithm
        self.memory = collections.deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = nn.Sequential(
            nn.Linear(state_size, 24), nn.ReLU(),
            nn.Linear(24, 48), nn.ReLU(),
            nn.Linear(48, action_size)
        ).to(DEVICE)
        self.opt = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done, next_action=None):
        self.memory.append((state, action, reward, next_state, done, next_action))

    def act(self, state):
        if self.policy == 'epsilon':
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)

        self.model.eval()
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
            q = self.model(s).cpu().numpy()

            if self.policy == 'softmax':
                probs = softmax(q)[0]
                return np.random.choice(self.action_size, p=probs)
            else:
                return int(np.argmax(q[0]))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        state_b = torch.as_tensor(np.vstack([m[0] for m in minibatch]), dtype=torch.float32, device=DEVICE)
        action_b = torch.as_tensor([m[1] for m in minibatch], dtype=torch.long, device=DEVICE)
        reward_b = torch.as_tensor([m[2] for m in minibatch], dtype=torch.float32, device=DEVICE)
        next_state_b = torch.as_tensor(np.vstack([m[3] for m in minibatch]), dtype=torch.float32, device=DEVICE)
        done_b = torch.as_tensor([m[4] for m in minibatch], dtype=torch.float32, device=DEVICE)

        self.model.eval()
        with torch.no_grad():
            if self.algorithm == 'qlearning':
                next_q_max = self.model(next_state_b).max(dim=1).values
            else:  # SARSA
                next_actions = torch.as_tensor([m[5] for m in minibatch], dtype=torch.long, device=DEVICE)
                next_q = self.model(next_state_b)
                next_q_max = next_q[torch.arange(batch_size), next_actions]

            target_scalar = reward_b + self.gamma * next_q_max * (1.0 - done_b)
            target_full = self.model(state_b).clone()
            target_full[torch.arange(batch_size), action_b] = target_scalar

        self.model.train()
        pred_q = self.model(state_b)
        loss = self.loss_fn(pred_q, target_full)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(agent, env, episodes=200):
    episode_rewards = []
    avg_rewards = collections.deque(maxlen=50)

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        total_reward = 0

        for time in range(200):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            next_state = np.reshape(next_state, [1, agent.state_size])

            if agent.algorithm == 'sarsa':
                next_action = agent.act(next_state)
                agent.remember(state, action, reward, next_state, done, next_action)
            else:
                agent.remember(state, action, reward, next_state, done)

            state = next_state
            if done:
                break
            if len(agent.memory) > 32:
                agent.replay(32)

        avg_rewards.append(total_reward)
        episode_rewards.append(np.mean(avg_rewards))

    return episode_rewards

# 训练4个agent
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agents = {
    'Q-learning + ε-greedy': DQNAgent(state_size, action_size, 'epsilon', 'qlearning'),
    'Q-learning + Softmax': DQNAgent(state_size, action_size, 'softmax', 'qlearning'),
    'SARSA + ε-greedy': DQNAgent(state_size, action_size, 'epsilon', 'sarsa'),
    'SARSA + Softmax': DQNAgent(state_size, action_size, 'softmax', 'sarsa')
}

results = {}
for name, agent in agents.items():
    print(f"训练 {name}...")
    rewards = train_agent(agent, env, episodes=200)
    results[name] = rewards

# Task 8.1: 绘制平均奖励曲线
plt.figure(figsize=(12, 6))
for name, rewards in results.items():
    plt.plot(rewards, label=name)
plt.xlabel('训练轮次')
plt.ylabel('最近50轮平均奖励')
plt.title('不同算法和策略的性能对比')
plt.legend()
plt.grid(True)
plt.savefig('Task8_1_rewards.png', dpi=300, bbox_inches='tight')
print("图表已保存: Task8_1_rewards.png")

# 额外对比图：最终性能对比
final_perf = {name: rewards[-1] for name, rewards in results.items()}
plt.figure(figsize=(10, 6))
bars = plt.bar(final_perf.keys(), final_perf.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.xlabel('算法')
plt.ylabel('最终平均奖励')
plt.title('不同算法的最终性能对比')
plt.xticks(rotation=15, ha='right')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('Task8_1_final_performance.png', dpi=300, bbox_inches='tight')
print("图表已保存: Task8_1_final_performance.png")
print("\n=== Task 08 完成 ===")


