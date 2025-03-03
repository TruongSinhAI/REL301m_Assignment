import pygame
import random
from src.constants.config import *
from src.agents.frog import *
from src.agents.vehicle import *
from src.utils.resource_manager import ResourceManager
from src.env.env import *

def simple_ai_policy(agent, vehicles): # (giữ nguyên)
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

def load_model(filename, state_dim, action_dim):
    """Load model đã lưu và khởi tạo lại agent"""
    from src.agents.td_agent import A2CAgentCNN  # Đảm bảo import đúng
    agent = A2CAgentCNN(state_dim, action_dim, hidden_dim=256, lr=0.001, gamma=0.9)
    agent.model.load_state_dict(torch.load(filename))
    agent.model.eval()
    return agent
def main(): # (chỉnh sửa để dừng âm thanh động cơ khi thoát game)
    env = CarCrossingEnv()
    running = True
    mouse_pos = (0, 0)
    observation = env.reset()
    agent_ai = load_model("a2c_agent_final.pth", 2, 5)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p and env.game_state == GameState.PLAYING:
                    env.game_state = GameState.PAUSED
                    # **Dừng âm thanh động cơ khi pause (nếu có)**
                    # env.stop_car_engine_sound()
                elif event.key == pygame.K_p and env.game_state == GameState.PAUSED:
                    env.game_state = GameState.PLAYING
                    # **Tiếp tục phát âm thanh động cơ khi unpause (nếu có)**
                    # env.play_car_engine_sound()
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
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(observation).unsqueeze(0)
                    policy_logits, _ = agent_ai.model(state_tensor)
                    action = torch.argmax(policy_logits, dim=1).item()

                # action = simple_ai_policy(env.agent, env.vehicles)

            observation, reward, done, info = env.step(action)

            if done:
                env.game_state = GameState.GAME_OVER
                # **Dừng âm thanh động cơ khi game over (nếu có)**
                # env.stop_car_engine_sound()

        env.render(env.mode)

    # **Dừng âm thanh động cơ khi thoát game (nếu có)**
    # env.stop_car_engine_sound()
    env.close()

if __name__ == "__main__":
    main()