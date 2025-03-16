import torch
import torch.nn as nn
import torch.optim as optim

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(input_dim, 128)

        # Actor
        self.actor = nn.Linear(128, action_dim)

        # Critic
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, epochs=10):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.action_dim = action_dim
        self.policy_net = self.model  # For compatibility with test function

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action_probs, value = self.model(state)
        action = torch.multinomial(action_probs, 1).item()
        log_prob = torch.log(action_probs[action])
        return action, log_prob, value

    def update(self, rewards, log_probs, values, old_log_probs):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns).to(device)
        log_probs = torch.stack(log_probs).to(device)
        values = torch.stack(values).to(device).squeeze()
        old_log_probs = torch.stack(old_log_probs).to(device)

        advantage = returns - values

        for _ in range(self.epochs):
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            actor_loss = -torch.mean(torch.min(ratio * advantage, clipped_ratio * advantage))
            critic_loss = torch.mean((returns - values) ** 2)
            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # Compatibility method for the training function
    def optimize_model(self, log_probs, values, rewards, dones, old_log_probs=None):
        if old_log_probs is None:
            old_log_probs = log_probs  # If not provided, use current log probs
        self.update(rewards, log_probs, values, old_log_probs)