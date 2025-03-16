import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


class AdvancedActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dims=[256, 256]):
        super(AdvancedActorCritic, self).__init__()

        # Build shared feature layers
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.feature_extractor = nn.Sequential(*layers)

        # Actor head (policy)
        self.actor = nn.Linear(hidden_dims[-1], action_dim)

        # Critic head (value)
        self.critic = nn.Linear(hidden_dims[-1], 1)

        # Initialize with orthogonal initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        features = self.feature_extractor(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action_and_value(self, x, action=None):
        action_logits, value = self(x)
        dist = Categorical(logits=action_logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()

        return action, log_prob, entropy, value


class A2CAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            lr=3e-4,
            gamma=0.99,
            # Learning method parameters
            learning_method='td',  # 'td', 'nstep', 'mc', 'tdlambda'
            n_steps=5,  # For n-step TD
            td_lambda=0.95,  # For TD(λ)
            # Other hyperparameters
            entropy_coef=0.01,
            value_loss_coef=0.5,
            max_grad_norm=0.5
    ):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        # Learning method configuration
        self.learning_method = learning_method
        self.n_steps = n_steps
        self.td_lambda = td_lambda

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize network and optimizer
        self.network = AdvancedActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # For compatibility with the training code
        self.policy_net = self.network

    def select_action(self, state):
        """Select action for a single state"""
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(state)

        return action.cpu().item(), log_prob, value

    def compute_td_returns(self, rewards, values, dones, next_value=0):
        """Calculate 1-step TD returns"""
        returns = []
        advantages = []
        next_value = next_value

        for i in reversed(range(len(rewards))):
            # If done, there's no next value to consider
            next_val = 0 if dones[i] else next_value

            # TD target = r + γV(s')
            td_target = rewards[i] + self.gamma * next_val

            # Advantage = TD target - V(s)
            advantage = td_target - values[i].item()

            returns.insert(0, td_target)
            advantages.insert(0, advantage)

            # Update next_value for the previous time step
            next_value = values[i].item()

        return returns, advantages

    def compute_nstep_returns(self, rewards, values, dones, next_value=0):
        """Calculate n-step returns using bootstrapping"""
        returns = []
        advantages = []
        n_step_return = next_value

        for i in reversed(range(len(rewards))):
            # For terminal states, reset the return
            if dones[i]:
                n_step_return = 0

            # Add the current reward to the n-step return
            n_step_return = rewards[i] + self.gamma * n_step_return

            # Calculate advantage
            advantage = n_step_return - values[i].item()

            returns.insert(0, n_step_return)
            advantages.insert(0, advantage)

        return returns, advantages

    def compute_mc_returns(self, rewards, values, dones):
        """Calculate full Monte Carlo returns (no bootstrapping)"""
        returns = []
        G = 0

        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        # Calculate advantages as difference between returns and values
        advantages = [ret - val.item() for ret, val in zip(returns, values)]

        return returns, advantages

    def compute_tdlambda_returns(self, rewards, values, dones, next_value=0):
        """Calculate TD(λ) returns using eligibility traces"""
        returns = []
        advantages = []
        gae = 0

        # Append next_value for proper bootstrapping
        value_buffer = values + [next_value]

        for i in reversed(range(len(rewards))):
            # For terminal states, there's no next value
            next_val = 0 if dones[i] else value_buffer[i + 1]

            # TD error
            delta = rewards[i] + self.gamma * next_val - value_buffer[i]

            # Generalized Advantage Estimation with λ as eligibility trace decay
            gae = delta + self.gamma * self.td_lambda * (1 - dones[i]) * gae

            # Return = advantage + value
            returns.insert(0, gae + value_buffer[i])
            advantages.insert(0, gae)

        return returns, advantages

    def compute_returns_and_advantages(self, rewards, values, dones, next_value=0):
        """Compute returns and advantages based on the chosen learning method"""
        if self.learning_method == 'td':
            # 1-step TD learning (Bellman equation)
            return self.compute_td_returns(rewards, values, dones, next_value)

        elif self.learning_method == 'nstep':
            # n-step TD learning
            return self.compute_nstep_returns(rewards, values, dones, next_value)

        elif self.learning_method == 'mc':
            # Pure Monte Carlo returns
            return self.compute_mc_returns(rewards, values, dones)

        elif self.learning_method == 'tdlambda':
            # TD(λ) with eligibility traces
            return self.compute_tdlambda_returns(rewards, values, dones, next_value)

        else:
            raise ValueError(f"Unknown learning method: {self.learning_method}")

    def update(self, states, actions, log_probs, values, rewards, dones, next_value=None):
        """Update policy and value networks using collected experience"""
        if next_value is None:
            next_value = 0

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.stack(log_probs).to(self.device)

        # Compute returns and advantages using the specified learning method
        returns, advantages = self.compute_returns_and_advantages(
            rewards, values, dones, next_value)

        # Convert to tensors
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Optional: normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get new action log probs and values
        _, new_log_probs, entropy, new_values = self.network.get_action_and_value(states, actions)

        # Calculate policy loss
        policy_loss = -(new_log_probs * advantages).mean()

        # Calculate value loss
        value_loss = F.mse_loss(new_values.squeeze(), returns)

        # Total loss with entropy regularization
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # Return metrics for monitoring
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_return': returns.mean().item()
        }

    def optimize_model(self, log_probs, values, rewards, dones, states=None, actions=None, next_value=None):
        """Compatibility method for the training function"""
        if states is None or actions is None:
            raise ValueError("States and actions must be provided for optimization")

        return self.update(states, actions, log_probs, values, rewards, dones, next_value)