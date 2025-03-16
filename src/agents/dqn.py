import torch
import torch.optim as optim
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.nn.functional as F

# Chọn device: GPU nếu có, ngược lại dùng CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        # Value stream
        self.value_fc = nn.Linear(128, 1)
        # Advantage stream
        self.advantage_fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        # Tính Q = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=300):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Các tham số epsilon cho epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Khởi tạo policy và target network theo kiến trúc Dueling DQN
        self.policy_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=100000)
        self.batch_size = 256

    def select_action(self, state, evaluate=False):
        """
        Nếu evaluate=True thì sử dụng policy mũi tên (greedy) – không có yếu tố khám phá.
        """
        if evaluate:
            # Chế độ đánh giá: chọn hành động tốt nhất theo Q-value (không random)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
        else:
            # Chế độ training: áp dụng epsilon-greedy
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
        self.memory.append((state, action, reward, next_state, done))

    def optimize_model(self):
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
        # Sử dụng Double DQN: chọn hành động tối ưu từ policy_net, tính Q-target từ target_net
        next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
        next_q = self.target_net(next_states).gather(1, next_actions)
        expected_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
