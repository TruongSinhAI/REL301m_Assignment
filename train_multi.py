
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from src.agents.dqn import DQNAgent

def train_agents(env, agents, num_episodes=500, target_update=10):
    num_agents = env.num_agents  # Số lượng agent trong môi trường
    best_rewards = [-float('inf')] * num_agents  # Lưu phần thưởng tốt nhất của từng agent

    for episode in range(num_episodes):
        # Reset môi trường trả về danh sách trạng thái (mỗi agent một trạng thái)
        states = env.reset()
        total_rewards = [0] * num_agents
        done = False
        step_count = 0

        while not done and step_count < 500:
            # Mỗi agent chọn hành động dựa trên trạng thái của nó
            actions = [agents[i].select_action(states[i]) for i in range(num_agents)]
            # Thực hiện bước environment: trả về next_states, rewards, done và thông tin bổ sung
            next_states, rewards, done, _ = env.step(actions)
            # Lưu transition và cộng dồn phần thưởng cho từng agent
            for i in range(num_agents):
                agents[i].store_transition(states[i], actions[i], rewards[i], next_states[i], done)
                total_rewards[i] += rewards[i]
            # Cập nhật mô hình cho từng agent
            for i in range(num_agents):
                agents[i].optimize_model()
            states = next_states
            step_count += 1

        # Cập nhật target network định kỳ
        if episode % target_update == 0:
            for i in range(num_agents):
                agents[i].update_target_network()

        # Lưu lại model nếu agent đạt phần thưởng tốt nhất trong episode này
        for i in range(num_agents):
            if total_rewards[i] > best_rewards[i]:
                best_rewards[i] = total_rewards[i]
                model_path = f"dqn_agent_{i}_best.pth"
                torch.save(agents[i].policy_net.state_dict(), model_path)
                print(f"Agent {i}: Best model saved as {model_path}")

        avg_reward = sum(total_rewards) / num_agents
        print(f"Episode {episode}: Avg Total Reward = {avg_reward}, Steps = {step_count} - Scores = {env.scores}")

    print("Training completed.")

def test_agents(env, agents, num_episodes=10):
    total_rewards = [0] * env.num_agents
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for episode in range(num_episodes):
        states = env.reset()
        episode_rewards = [0] * env.num_agents
        done = False
        while not done:
            actions = []
            for i in range(env.num_agents):
                state_tensor = torch.FloatTensor(states[i]).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = agents[i].policy_net(state_tensor)
                actions.append(q_values.argmax().item())
            next_states, rewards, done, _ = env.step(actions)
            for i in range(env.num_agents):
                episode_rewards[i] += rewards[i]
            states = next_states

        for i in range(env.num_agents):
            total_rewards[i] += episode_rewards[i]
        print(f"Test Episode {episode}: Rewards = {episode_rewards} - Scores = {env.scores}")

    avg_rewards = [r / num_episodes for r in total_rewards]
    print("Average Test Rewards per agent:", avg_rewards)

if __name__ == "__main__":
    # Giả sử môi trường đã được định nghĩa theo phiên bản multi-agent (xem code môi trường đã chỉnh sửa ở phần trước)
    from src.env.multi_agent_env import BaseCarCrossingEnv  # Điều chỉnh import cho phù hợp với dự án của bạn

    env = BaseCarCrossingEnv()  # Môi trường không GUI để training nhanh hơn
    # Theo get_observation của mỗi agent, state vector có kích thước: 4 + 15 = 19
    state_dim = 19
    action_dim = 5  # 5 hành động

    # Tạo danh sách các agent độc lập (mỗi agent có mô hình riêng)
    num_agents = env.num_agents  # Số agent được thiết lập trong môi trường (hoặc bạn có thể tự đặt)
    agents = [DQNAgent(state_dim, action_dim) for _ in range(num_agents)]

    # Huấn luyện các agent
    train_agents(env, agents, num_episodes=10000, target_update=10)

    # Lưu model cuối cùng của từng agent sau khi training
    for i, agent in enumerate(agents):
        model_path = f"dqn_agent_{i}.pth"
        torch.save(agent.policy_net.state_dict(), model_path)
        print(f"Agent {i} model saved as {model_path}")

    # Tải model cho việc test (mỗi agent tải model riêng của mình)
    loaded_agents = [DQNAgent(state_dim, action_dim) for _ in range(num_agents)]
    for i in range(num_agents):
        model_path = f"dqn_agent_{i}.pth"
        loaded_agents[i].policy_net.load_state_dict(torch.load(model_path))
        loaded_agents[i].policy_net.eval()  # Chuyển sang chế độ đánh giá
        print(f"Loaded model for agent {i} from {model_path}")

    # Test các agent đã học
    test_agents(env, loaded_agents, num_episodes=10)

    env.close()
