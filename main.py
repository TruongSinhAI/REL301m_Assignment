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
from src.env.env import CarCrossingEnv, GameState


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
    # Sử dụng 3 đặc trưng đầu tiên: agent_x_norm, agent_y_norm, lane_norm
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


# Hàm chọn action cho các thuật toán tabular (ví dụ SARSA)
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


# Hàm load các giá trị tabular đã lưu (ví dụ Q-values của SARSA)
def load_tabular_values(filename):
    with open(filename, "rb") as f:
        values = pickle.load(f)
    return values


def main():
    # Chọn thuật toán cần sử dụng: "dqn", "sarsa", "simple" hoặc "a2c" (nếu có)
    ALGO_CHOICE = "dqn"  # Thay đổi giá trị này để đánh giá thuật toán khác

    env = CarCrossingEnv()
    running = True
    mouse_pos = (0, 0)
    observation = env.reset()

    # Khởi tạo agent hoặc tải kết quả theo thuật toán đã chọn
    agent_ai = None
    tabular_Q = None

    if ALGO_CHOICE == "dqn":
        model_path = "dqn_agent_best.pth"
        agent_ai = DQNAgent(19, 5)
        agent_ai.epsilon = 0
        agent_ai.policy_net.load_state_dict(torch.load(model_path, weights_only=True))
        agent_ai.policy_net.eval()
        print(f"Loaded DQN model from {model_path}")
    elif ALGO_CHOICE == "sarsa":
        q_path = "V_td.pkl"
        tabular_Q = load_tabular_values(q_path)
        print(f"Loaded SARSA Q-values from {q_path}")
    elif ALGO_CHOICE == "a2c":
        model_path = "a2c_agent_final.pth"
        agent_ai = load_model(model_path, 2, 5)  # Lưu ý: state_dim và action_dim tùy thuộc vào model A2C của bạn
        print(f"Loaded A2C model from {model_path}")
    else:
        # Nếu sử dụng simple policy
        print("Using simple AI policy.")

    # Vòng lặp chính game
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p and env.game_state == GameState.PLAYING:
                    env.game_state = GameState.PAUSED
                    # Dừng âm thanh động cơ nếu có
                elif event.key == pygame.K_p and env.game_state == GameState.PAUSED:
                    env.game_state = GameState.PLAYING
                elif event.key == pygame.K_r and env.game_state == GameState.GAME_OVER:
                    env.restart_game()
                elif event.key == pygame.K_m and env.game_state == GameState.GAME_OVER:
                    env.return_to_menu()
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
                    if env.game_state == GameState.MENU:
                        for button in env.menu_buttons:
                            if button.check_click(mouse_pos):
                                break
                    elif env.game_state == GameState.GAME_OVER:
                        for button in env.game_over_buttons:
                            if button.check_click(mouse_pos):
                                break

        if env.game_state == GameState.PLAYING:
            # Lựa chọn hành động theo chế độ được chọn
            if env.mode == "manual":
                keys = pygame.key.get_pressed()
                action = 0
                if keys[pygame.K_UP]:
                    action = 1
                elif keys[pygame.K_DOWN]:
                    action = 2
                elif keys[pygame.K_LEFT]:
                    action = 3
                elif keys[pygame.K_RIGHT]:
                    action = 4
            else:
                if ALGO_CHOICE == "dqn" or ALGO_CHOICE == "a2c":
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q_values = agent_ai.policy_net(state_tensor)
                    action = q_values.argmax().item()
                elif ALGO_CHOICE == "sarsa":
                    # Sử dụng bảng Q từ SARSA và hàm discretize_state để chọn hành động
                    action = tabular_policy(observation, tabular_Q, env.action_space)
                else:
                    # Sử dụng simple AI policy
                    action = simple_ai_policy(env.agent, env.vehicles)

            observation, reward, done, info = env.step(action)
            if done:
                env.game_state = GameState.GAME_OVER

        env.render(env.mode)
    # Khi thoát game, dừng âm thanh nếu có và đóng môi trường
    env.close()


if __name__ == "__main__":
    main()
