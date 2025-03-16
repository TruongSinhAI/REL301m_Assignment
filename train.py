# from src.agents.td_agent import A2CAgentCNN  # Giả sử bạn đã triển khai A2CAgent như hướng dẫn trước
# from src.env.env import BaseCarCrossingEnv
# import matplotlib.pyplot as plt
# import numpy as np
# import time
# import os
# import torch
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# def train_agent(num_episodes=2000, max_steps_per_episode=1000, save_interval=100, evaluation_interval=50):
#     # Initialize environment
#     env = BaseCarCrossingEnv()
#
#     # Get correct state dimensions from environment observation
#     observation = env.reset()
#     input_channels = observation.shape[0]  # Should be 1 for your grid representation
#     print(f"Observation shape: {observation.shape}")
#
#     action_dim = 5  # Actions: no move, up, down, left, right
#
#     # Create agent with appropriate parameters
#     agent = A2CAgentCNN(
#         input_channels=input_channels,
#         action_dim=action_dim,
#         hidden_dim=256,  # Larger network for more complex policies
#         lr=0.00003,  # Slightly lower learning rate for stability
#         gamma=0.9,  # Discount factor
#         entropy_coef=0.1  # Entropy coefficient to encourage exploration
#     )
#
#     # Prepare data collection
#     episode_rewards = []
#     episode_scores = []
#     episode_steps = []
#     win_count = 0
#     win_rates = []
#     avg_losses = []
#
#     # Implement early stopping
#     best_win_rate = 0
#     patience = 50  # Number of episodes to wait before early stopping
#     no_improvement_count = 0
#
#     print("Starting training...")
#     # Biến lưu điểm cao nhất từng đạt được
#     best_score = float('-inf')
#     for episode in range(1, num_episodes + 1):
#         observation = env.reset()
#         total_reward = 0
#         steps = 0
#         trajectory = []
#         episode_win = False
#         actor_losses = []
#         critic_losses = []
#         entropies = []
#
#
#         # Epsilon-greedy exploration, decreasing over time
#         epsilon = max(0.05, 1.0 - episode / (num_episodes * 0.7))
#
#         for step in range(max_steps_per_episode):
#             # With probability epsilon, choose random action
#             if np.random.random() < epsilon:
#                 action = np.random.randint(0, action_dim)
#             else:
#                 action = agent.select_action(observation)
#
#             next_observation, reward, done, info = env.step(action)
#
#             # Reward shaping: more sophisticated rewards based on game state
#             # Store trajectory with original reward for training
#             trajectory.append((observation, action, reward, next_observation, done))
#
#             total_reward += reward
#             steps += 1
#             observation = next_observation
#
#             if done:
#                 # Check win condition
#                 if env.score >= 120:
#                     episode_win = True
#                     win_count += 1
#                 break
#
#         # Update agent on trajectory data
#         if len(trajectory) > 0:
#             actor_loss, critic_loss, entropy = agent.update(trajectory)
#             actor_losses.append(actor_loss)
#             critic_losses.append(critic_loss)
#             entropies.append(entropy)
#
#         # Record metrics
#         episode_rewards.append(total_reward)
#         episode_scores.append(env.score)
#         episode_steps.append(steps)
#         win_rates.append(win_count / episode)
#
#         if len(actor_losses) > 0:
#             avg_losses.append((np.mean(actor_losses), np.mean(critic_losses), np.mean(entropies)))
#
#         # Learning rate decay
#         if episode % 200 == 0:
#             for param_group in agent.optimizer.param_groups:
#                 param_group['lr'] = max(param_group['lr'] * 0.9, 0.00005)
#             print(f"Learning rate adjusted to {agent.optimizer.param_groups[0]['lr']}")
#
#         # Print progress
#         if episode % 10 == 0:
#             avg_reward = np.mean(episode_rewards[-10:])
#             avg_score = np.mean(episode_scores[-10:])
#             recent_wins = sum(1 for score in episode_scores[-10:] if score >= 120)
#             recent_win_rate = recent_wins / 10.0
#
#             print(f"Episode {episode}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Avg Score: {avg_score:.2f}")
#             print(f"Total Wins: {win_count}, Win Rate: {win_rates[-1]:.4f}, Recent Win Rate: {recent_win_rate:.2f}")
#             print(f"Steps: {steps}, Epsilon: {epsilon:.3f}")
#
#             if len(avg_losses) > 0:
#                 recent_losses = avg_losses[-1]
#                 print(
#                     f"Actor Loss: {recent_losses[0]:.4f}, Critic Loss: {recent_losses[1]:.4f}, Entropy: {recent_losses[2]:.4f}")
#
#             print("-" * 50)
#
#         # Save model and plot performance
#         if episode % save_interval == 0:
#             save_model(agent, f"a2c_agent_episode_{episode}.pth")
#             plot_performance(episode_rewards, episode_scores, win_rates)
#
#         # Evaluate agent
#         if episode % evaluation_interval == 0:
#             eval_win_rate = evaluate_agent(agent, num_episodes=20)
#             print(f"Evaluation after {episode} episodes: Win Rate = {eval_win_rate:.4f}")
#
#             # Early stopping check
#             if eval_win_rate > best_win_rate:
#                 best_win_rate = eval_win_rate
#                 no_improvement_count = 0
#                 # Save best model
#                 save_model(agent, "a2c_agent_best.pth")
#             else:
#                 no_improvement_count += 1
#
#             if no_improvement_count >= patience:
#                 print(f"No improvement for {patience} evaluations. Early stopping.")
#                 break
#
#
#
#         # Sau khi mỗi episode kết thúc
#         if env.score > best_score:
#             best_score = env.score
#             best_episode = episode
#             save_model(agent, "a2c_agent_best.pth")
#             print(f"New best model saved at episode {best_episode} with score {best_score:.2f}")
#
#     # Save final model
#     save_model(agent, "a2c_agent_final.pth")
#     plot_performance(episode_rewards, episode_scores, win_rates)
#     return agent, episode_rewards, episode_scores, win_rates
#
#
# def evaluate_agent(agent, num_episodes=20, render=False):
#     """Evaluate agent without exploration"""
#     env = BaseCarCrossingEnv()
#     wins = 0
#
#     for episode in range(num_episodes):
#         observation = env.reset()
#         done = False
#
#         while not done:
#             # Use deterministic actions for evaluation
#             action = agent.select_action(observation, deterministic=True)
#             next_observation, reward, done, _ = env.step(action)
#             observation = next_observation
#
#             if render:
#                 env.render(mode="ai")
#                 time.sleep(0.05)
#
#         # Check win condition
#         if env.score >= 120:
#             wins += 1
#
#     win_rate = wins / num_episodes
#     return win_rate
#
#
# def save_model(agent, filename):
#     """Lưu lại state_dict của mô hình Actor-Critic"""
#     torch.save(agent.model.state_dict(), filename)
#     print(f"Model saved to {filename}")
#
#
# def load_model(filename, state_dim, action_dim):
#     """Load model đã lưu và khởi tạo lại agent"""
#     from src.agents.td_agent import A2CAgentCNN  # Đảm bảo import đúng
#     agent = A2CAgentCNN(state_dim, action_dim, hidden_dim=128, lr=0.001, gamma=0.9)
#     agent.model.load_state_dict(torch.load(filename))
#     agent.model.eval()
#     return agent
#
#
# def plot_performance(rewards, scores, win_rates):
#     """Vẽ các số liệu hiệu năng huấn luyện"""
#     plt.figure(figsize=(15, 15))
#
#     # Vẽ tổng reward của mỗi episode
#     plt.subplot(4, 1, 1)
#     plt.plot(rewards)
#     plt.title('Episode Rewards')
#     plt.xlabel('Episode')
#     plt.ylabel('Total Reward')
#
#     # Vẽ moving average của reward
#     plt.subplot(4, 1, 2)
#     window_size = min(100, len(rewards))
#     if window_size > 0:
#         moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
#         plt.plot(moving_avg)
#         plt.title(f'Moving Average of Rewards (Window Size: {window_size})')
#         plt.xlabel('Episode')
#         plt.ylabel('Average Reward')
#
#     # Vẽ score của mỗi episode
#     plt.subplot(4, 1, 3)
#     plt.plot(scores)
#     plt.title('Episode Scores')
#     plt.xlabel('Episode')
#     plt.ylabel('Score')
#
#     # Vẽ tỷ lệ thắng tích lũy
#     plt.subplot(4, 1, 4)
#     plt.plot(win_rates)
#     plt.title('Cumulative Win Rate')
#     plt.xlabel('Episode')
#     plt.ylabel('Win Rate')
#     plt.ylim([0, 1])
#
#     plt.tight_layout()
#     plt.savefig('training_performance.png')
#     plt.close()
#
#
# def test_agent(agent, num_episodes=10, render=True):
#     """Kiểm tra agent đã huấn luyện"""
#     env = BaseCarCrossingEnv()
#
#     if render:
#         try:
#             from src.env.env import CarCrossingEnv
#             env = CarCrossingEnv()
#         except ImportError:
#             print("Cannot import CarCrossingEnv, using BaseCarCrossingEnv instead")
#             render = False
#
#     total_rewards = []
#     total_scores = []
#     wins = 0
#
#     for episode in range(num_episodes):
#         observation = env.reset()
#         total_reward = 0
#         done = False
#         steps = 0
#
#         while not done and steps <= 1000:
#             # Sử dụng chính sách xác định (deterministic) cho test: chọn hành động có xác suất cao nhất
#             with torch.no_grad():
#                 state_tensor = torch.FloatTensor(observation).unsqueeze(0)
#                 policy_logits, _ = agent.model(state_tensor)
#                 action = torch.argmax(policy_logits, dim=1).item()
#
#             next_observation, reward, done, _ = env.step(action)
#             total_reward += reward
#             observation = next_observation
#             steps += 1
#
#             if render:
#                 env.render(mode="ai")
#                 time.sleep(0.05)
#
#             if done and env.score > 120:
#                 wins += 1
#
#         total_rewards.append(total_reward)
#         total_scores.append(env.score)
#         win_status = "WIN" if reward > 0 else "LOSS"
#         print(f"Test Episode {episode+1}: {win_status}, Reward = {total_reward:.2f}, Score = {env.score}")
#
#     avg_reward = sum(total_rewards) / num_episodes
#     avg_score = sum(total_scores) / num_episodes
#     win_rate = wins / num_episodes
#     print(f"Average Test Reward: {avg_reward:.2f}, Average Score: {avg_score:.2f}")
#     print(f"Test Win Rate: {win_rate:.2f} ({wins}/{num_episodes})")
#
#     if render:
#         env.close()
#
#
# if __name__ == "__main__":
#     os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
#     # Huấn luyện agent với số episode mong muốn
#     agent, rewards, scores, win_rates = train_agent(num_episodes=500, max_steps_per_episode=500, save_interval=100000)
#     # Test agent đã huấn luyện
#     test_agent(agent, num_episodes=10, render=False)