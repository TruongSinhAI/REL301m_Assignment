import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch  # Sau đó import torch hoặc thư viện bạn đang dùng

from src.agents.dqn import DQNAgent

def train_agent(env, agent, num_episodes=500, target_update=10):
    min_reward = -float('inf')
    for episode in range(num_episodes):
        state = env.reset()  # Observation vector (ví dụ: kích thước 19)
        total_reward = 0
        done = False
        step_count = 0
        while not done and step_count<500:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.optimize_model()
            state = next_state
            total_reward += reward
            step_count += 1

        if episode % target_update == 0:
            agent.update_target_network()
        if total_reward > min_reward:
            min_reward = total_reward
            model_path = "dqn_agent_best.pth"
            torch.save(agent.policy_net.state_dict(), model_path)
            print(f"Model saved as {model_path}")
        print(f"Episode {episode}: Total Reward = {total_reward}, Steps = {step_count} - Score = {env.score}")
    print("Training completed.")

def test_agent(env, agent, num_episodes=10):
    total_rewards = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_score = 0

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        done = False
        while not done:
            # Sử dụng policy net để chọn hành động (không dùng epsilon-greedy)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = agent.policy_net(state_tensor)
            action = q_values.argmax().item()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
        total_score += env.score
        total_rewards.append(total_reward)
        print(f"Test Episode {episode}: Total Reward = {total_reward} - Total Score = {env.score}")
    avg_reward = sum(total_rewards) / num_episodes
    avg_score = total_score / num_episodes
    print(f"Average Test Reward: {avg_reward} - Average Test Score: {avg_score}")

if __name__ == "__main__":
    # Giả sử môi trường đã được định nghĩa ở code trước
    from src.env.env import CarCrossingEnv  # Điều chỉnh import cho đúng với dự án của bạn

    env = CarCrossingEnv()  # Hoặc BaseCarCrossingEnv() nếu bạn không cần GUI trong train
    # Theo code get_observation đã cải tiến, state vector có kích thước: 4 (agent + risk) + 15 (5 xe x 3 đặc trưng) = 19
    state_dim = 19
    action_dim = 5  # 5 hành động

    # Khởi tạo agent
    agent = DQNAgent(state_dim, action_dim)

    # Huấn luyện agent
    train_agent(env, agent, num_episodes=10000, target_update=10)

    # Lưu model sau khi train
    model_path = "dqn_agent.pth"
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"Model saved as {model_path}")

    # Tái sử dụng model: Khởi tạo agent mới và load trọng số đã lưu
    agent_loaded = DQNAgent(state_dim, action_dim)
    agent_loaded.policy_net.load_state_dict(torch.load(model_path))
    agent_loaded.policy_net.eval()  # Chuyển sang chế độ đánh giá (inference)
    print(f"Loaded model from {model_path}")

    # Kiểm tra agent đã học (sử dụng model đã load)
    test_agent(env, agent_loaded, num_episodes=10)

    env.close()
