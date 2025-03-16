import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pygame
import torch
import pickle
import random
import numpy as np

from src.agents.dqn import DQNAgent
from src.constants.config import *
from src.agents.frog import *
from src.agents.vehicle import *
from src.utils.resource_manager import ResourceManager
from src.env.multi_agent_env import MultiAgentCarCrossingEnvGUI, GameState


# Hàm simple_ai_policy giữ nguyên (chọn action an toàn theo thứ tự ưu tiên)
def simple_ai_policy(agent, vehicles):
    actions = [1, 3, 4, 2, 0]
    for action in actions:
        new_rect = agent.get_rect().copy()
        dx, dy = 0, 0
        if action == 1:
            dy = -agent.speed
        elif action == 2:
            dy = agent.speed
        elif action == 3:
            dx = -agent.speed
        elif action == 4:
            dx = agent.speed
        new_rect.x += dx
        new_rect.y += dy
        safe = True
        for v in vehicles:
            if new_rect.colliderect(v.get_rect()):
                safe = False
                break
        if safe:
            return action
    return 0


# Hàm load model của một thuật toán khác (ví dụ A2C, nếu có)
def load_model(filename, state_dim, action_dim):
    from src.agents.td_agent import A2CAgentCNN  # Đảm bảo import đúng
    agent = A2CAgentCNN(state_dim, action_dim, hidden_dim=256, lr=0.001, gamma=0.9)
    agent.model.load_state_dict(torch.load(filename))
    agent.model.eval()
    return agent


# Hàm discretize state (cho các thuật toán tabular)
def discretize_state(state, bins=(10, 10, 5)):
    a_x = state[0]
    a_y = state[1]
    lane = state[2]
    dx = int(a_x * bins[0])
    dy = int(a_y * bins[1])
    dlane = int(lane * bins[2])
    dx = min(dx, bins[0] - 1)
    dy = min(dy, bins[1] - 1)
    dlane = min(dlane, bins[2] - 1)
    return (dx, dy, dlane)


# Hàm chọn action cho thuật toán tabular (ví dụ SARSA)
def tabular_policy(observation, Q, action_space):
    d_state = discretize_state(observation)
    best_action = None
    best_val = -float("inf")
    for a in action_space:
        val = Q.get((d_state, a), 0.0)
        if val > best_val:
            best_val = val
            best_action = a
    if best_action is None:
        best_action = random.choice(action_space)
    return best_action


# Hàm load bảng giá trị tabular (ví dụ Q-values của SARSA)
def load_tabular_values(filename):
    with open(filename, "rb") as f:
        values = pickle.load(f)
    return values


def main():
    # Initialize pygame
    pygame.init()

    # Khởi tạo môi trường (giả sử đã được chuyển sang multi-agent và hỗ trợ GUI)
    env = MultiAgentCarCrossingEnvGUI()
    running = True
    # Lấy danh sách observation cho các agent (mỗi agent có state vector riêng)
    observation = env.reset()

    # Ban đầu chưa có lựa chọn nào được chọn
    ALGO_CHOICE = None
    agents = None  # Dành cho các thuật toán AI (DQN, A2C)
    tabular_Q = None  # Dành cho SARSA
    env.mode = "manual"  # Mặc định chế độ manual
    # Sử dụng font để hiển thị menu lựa chọn
    font = pygame.font.SysFont("Helvetica", 24)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Nếu đang ở trạng thái MENU, cho phép người dùng chọn option bằng phím số
                if env.game_state == GameState.MENU:
                    if event.key == pygame.K_1:
                        ALGO_CHOICE = "manual"
                        env.mode = "manual"
                        print("Manual mode selected.")
                    elif event.key == pygame.K_2:
                        ALGO_CHOICE = "dqn"
                        env.mode = "ai"
                        # Tạo danh sách agent DQN cho mỗi agent trong môi trường
                        agents = []
                        for i in range(env.num_agents):
                            agent_dqn = DQNAgent(19, 5)
                            agent_dqn.epsilon = 0
                            model_path = f"dqn_agent_best.pth"
                            agent_dqn.policy_net.load_state_dict(torch.load(model_path, weights_only=True))
                            agent_dqn.policy_net.eval()
                            agent_dqn.epsilon = 0
                            agents.append(agent_dqn)
                        print("DQN mode selected.")
                    elif event.key == pygame.K_3:
                        ALGO_CHOICE = "sarsa"
                        env.mode = "ai"
                        tabular_Q = load_tabular_values("V_td.pkl")
                        print("SARSA mode selected.")
                    elif event.key == pygame.K_4:
                        ALGO_CHOICE = "a2c"
                        env.mode = "ai"
                        from src.agents.td_agent import A2CAgentCNN
                        agents = []
                        for i in range(env.num_agents):
                            agent_a2c = A2CAgentCNN(19, 5, hidden_dim=256, lr=0.001, gamma=0.9)
                            model_path = f"a2c_agent_{i}_best.pth"
                            agent_a2c.model.load_state_dict(torch.load(model_path))
                            agent_a2c.model.eval()
                            agents.append(agent_a2c)
                        print("A2C mode selected.")
                    elif event.key == pygame.K_5:
                        ALGO_CHOICE = "simple"
                        env.mode = "ai"
                        print("Simple AI mode selected.")
                    # Khi lựa chọn xong, chuyển trạng thái game từ MENU sang PLAYING
                    if ALGO_CHOICE is not None:
                        env.game_state = GameState.PLAYING
                else:
                    # Xử lý tạm dừng/tiếp tục game
                    if event.key == pygame.K_p:
                        if env.game_state == GameState.PLAYING:
                            env.game_state = GameState.PAUSED
                        elif env.game_state == GameState.PAUSED:
                            env.game_state = GameState.PLAYING
                    elif event.key == pygame.K_r and env.game_state == GameState.GAME_OVER:
                        env.restart_game()
                        observation = env.reset()
                    elif event.key == pygame.K_m and env.game_state == GameState.GAME_OVER:
                        env.return_to_menu()
                        ALGO_CHOICE = None
                        agents = None
                        tabular_Q = None
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos
                if env.game_state == GameState.MENU:
                    for button in env.menu_buttons:
                        button.check_hover(mouse_pos)
                elif env.game_state == GameState.GAME_OVER:
                    for button in env.game_over_buttons:
                        button.check_hover(mouse_pos)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    if env.game_state == GameState.MENU:
                        for button in env.menu_buttons:
                            if button.check_click(mouse_pos):
                                break
                    elif env.game_state == GameState.GAME_OVER:
                        for button in env.game_over_buttons:
                            if button.check_click(mouse_pos):
                                break

        # Nếu đang ở MENU, vẽ giao diện menu kèm các hướng dẫn lựa chọn
        if env.game_state == GameState.MENU:
            env.render(env.mode)
            menu_text = [
                "Press 1: Manual",
                "Press 2: DQN",
                "Press 3: SARSA",
                "Press 4: A2C",
                "Press 5: Simple AI"
            ]
            for idx, text in enumerate(menu_text):
                text_surface = font.render(text, True, (255, 255, 255))
                env.screen.blit(text_surface, (50, 50 + idx * 30))
            pygame.display.flip()
            continue

        # Nếu game đang PLAYING
        if env.game_state == GameState.PLAYING:
            actions = []
            # Nếu chưa có observation (hoặc sau reset) thì gọi lại reset
            if observation is None:
                observation = env.reset()
            # Với mỗi agent, chọn hành động tương ứng
            for i, obs in enumerate(observation):
                if env.mode == "manual":
                    # Ở chế độ manual: cho agent đầu tiên (i==0) dùng bàn phím, các agent còn lại mặc định action=0
                    if i == 0:
                        keys = pygame.key.get_pressed()
                        if keys[pygame.K_UP]:
                            act = 1
                        elif keys[pygame.K_DOWN]:
                            act = 2
                        elif keys[pygame.K_LEFT]:
                            act = 3
                        elif keys[pygame.K_RIGHT]:
                            act = 4
                        else:
                            act = 0
                    else:
                        act = 1
                    actions.append(act)
                else:
                    # Ở chế độ AI
                    if ALGO_CHOICE in ["dqn", "a2c"]:
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                        with torch.no_grad():
                            if ALGO_CHOICE == "dqn":
                                try:
                                    q_values = agents[i].policy_net(state_tensor)
                                except:
                                    pass
                            else:  # A2C, sử dụng model của agent_a2c
                                q_values = agents[i].model(state_tensor)
                        act = q_values.argmax().item()
                        actions.append(act)
                    elif ALGO_CHOICE == "sarsa":
                        act = tabular_policy(obs, tabular_Q, env.action_space)
                        actions.append(act)
                    elif ALGO_CHOICE == "simple":
                        act = simple_ai_policy(env.agents[i], env.vehicles)
                        actions.append(act)
            observation, rewards, done, info = env.step(actions)
            if done:
                env.game_state = GameState.GAME_OVER

        env.render(env.mode)
    env.close()


if __name__ == "__main__":
    main()