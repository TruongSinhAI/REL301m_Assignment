# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
#
#
# class ActorCriticCNN(nn.Module):
#     def __init__(self, input_channels, action_dim, hidden_dim=256):
#         super(ActorCriticCNN, self).__init__()
#
#         # Deeper CNN architecture
#         self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(64)
#
#         # Calculate flattened size
#         self.grid_width = 15  # SCREEN_WIDTH // cell_size
#         self.grid_height = 20  # SCREEN_HEIGHT // cell_size
#
#         conv_w = self.grid_width // 2
#         conv_h = self.grid_height // 2
#         conv_output_size = 64 * conv_w * conv_h
#         conv_output_size = 5120
#         # Separate networks for actor and critic
#         self.actor = nn.Sequential(
#             nn.Linear(conv_output_size, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, action_dim)
#         )
#
#         self.critic = nn.Sequential(
#             nn.Linear(conv_output_size, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, 1)
#         )
#
#         self._initialize_weights()
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
#                 nn.init.constant_(m.bias, 0.0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
#                 nn.init.constant_(m.bias, 0.0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, state):
#         x = F.relu(self.bn1(self.conv1(state)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = x.view(x.size(0), -1)
#
#         policy_logits = self.actor(x)
#         value = self.critic(x)
#         return policy_logits, value
#
#
# class A2CAgentCNN:
#     def __init__(self, input_channels, action_dim, hidden_dim=128, lr=1e-4, gamma=0.99, entropy_coef=0.1, epsilon=0.1):
#         self.gamma = gamma
#         self.entropy_coef = entropy_coef  # Tăng hệ số entropy để thúc đẩy khám phá
#         self.epsilon = epsilon            # Xác suất chọn hành động ngẫu nhiên
#         self.model = ActorCriticCNN(input_channels, action_dim, hidden_dim)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
#         self.action_dim = action_dim
#
#     def select_action(self, state, deterministic=False, temperature=1.0):
#         """
#         state: numpy array với shape (input_channels, height, width)
#         deterministic: nếu True thì chọn hành động tốt nhất; nếu False thì lấy mẫu từ phân phối
#         temperature: hệ số làm phẳng phân phối softmax
#         """
#         state = torch.FloatTensor(state).unsqueeze(0)
#         policy_logits, _ = self.model(state)
#
#         if deterministic:
#             action = torch.argmax(policy_logits, dim=1).item()
#         else:
#             # Với xác suất epsilon, chọn hành động ngẫu nhiên để tăng khám phá
#             if np.random.rand() < self.epsilon:
#                 action = np.random.choice(self.action_dim)
#             else:
#                 policy = F.softmax(policy_logits / temperature, dim=1)
#                 action_probs = policy.detach().cpu().numpy()[0]
#                 action = np.random.choice(self.action_dim, p=action_probs)
#
#         return action
#
#     def update(self, trajectory):
#         """
#         trajectory: danh sách các tuple (state, action, reward, next_state, done)
#         """
#         states, actions, rewards, next_states, dones = zip(*trajectory)
#
#         # Chuyển sang tensor
#         states = torch.FloatTensor(np.array(states))
#         actions = torch.LongTensor(actions).unsqueeze(1)
#         rewards = torch.FloatTensor(rewards)
#         dones = torch.FloatTensor(dones)
#
#         # Lấy các giá trị policy và value từ model
#         policy_logits, values = self.model(states)
#         values = values.squeeze()
#
#         # Tính log probabilities và entropy
#         log_probs = F.log_softmax(policy_logits, dim=1)
#         chosen_log_probs = log_probs.gather(1, actions).squeeze()
#         entropy = -(F.softmax(policy_logits, dim=1) * log_probs).sum(dim=1).mean()
#
#         # Tính target và advantage sử dụng GAE (ở đây đơn giản hóa thành target = reward + gamma * next_value)
#         with torch.no_grad():
#             _, next_values = self.model(torch.FloatTensor(np.array(next_states)))
#             next_values = next_values.squeeze()
#         targets = rewards + self.gamma * next_values * (1 - dones)
#         advantages = targets - values
#
#         # Tính loss cho actor và critic, cộng thêm entropy bonus
#         actor_loss = -(chosen_log_probs * advantages.detach()).mean()
#         critic_loss = F.smooth_l1_loss(values, targets.detach())
#         entropy_loss = -self.entropy_coef * entropy  # Khuyến khích khám phá
#
#         loss = actor_loss + critic_loss + entropy_loss
#
#         # Tối ưu
#         self.optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
#         self.optimizer.step()
#
#         return actor_loss.item(), critic_loss.item(), entropy.item()
