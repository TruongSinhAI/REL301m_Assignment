import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Chọn device: GPU nếu có, ngược lại dùng CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearQNetwork(nn.Module):
    """
    Mạng nơ-ron đơn giản với 1 lớp tuyến tính, dùng để xấp xỉ Q-value.
    """

    def __init__(self, input_dim, output_dim):
        super(LinearQNetwork, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class LinearDQNAgent:
    """
    Agent sử dụng hàm xấp xỉ tuyến tính để ước lượng Q-value.
    """

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=300):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Các tham số epsilon cho epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Khởi tạo policy và target network với kiến trúc tuyến tính
        self.policy_net = LinearQNetwork(state_dim, action_dim).to(device)
        self.target_net = LinearQNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

    def select_action(self, state, evaluate=False):
        """
        Chọn hành động theo epsilon-greedy. Nếu evaluate=True thì chỉ dùng policy_net.
        """
        if evaluate:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
        else:
            self.steps_done += 1
            eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
                            np.exp(-1. * self.steps_done / self.epsilon_decay)
            if random.random() < eps_threshold:
                return random.randrange(self.action_dim)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Lưu transition vào bộ nhớ replay."""
        self.memory.append((state, action, reward, next_state, done))

    def optimize_model(self):
        """Lấy mẫu từ replay buffer và cập nhật network."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        current_q = self.policy_net(states).gather(1, actions)
        # Double DQN: chọn hành động tối ưu theo policy_net, tính Q-target từ target_net
        next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
        next_q = self.target_net(next_states).gather(1, next_actions)
        expected_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Cập nhật target network bằng policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
