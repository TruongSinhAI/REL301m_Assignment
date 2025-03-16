import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pygame
import torch
import random
import numpy as np

from src.agents.dqn import DQNAgent
from src.constants.config import *
from src.agents.frog import *
from src.agents.vehicle import *
from src.utils.resource_manager import ResourceManager
from src.env.env import CarCrossingEnv, GameState


# Function for simple AI policy (selecting safe actions in priority order)
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


def main():
    # The three game modes we'll use:
    # "manual" - Manual play
    # "ai" - DQN agent plays
    # "coop" - Human and DQN cooperate (we'll handle this in the code logic)
    
    env = CarCrossingEnv()  # Initialize with default parameters
    
    # Modify the environment's modes to include only our 3 options
    # This assumes the environment has a property for available modes
    env.modes = ["manual", "ai", "coop"]  
    env.mode = "ai"  # Set the initial mode
    # If needed, update the menu buttons to reflect these three options
    # This would depend on how your menu is implemented
    # We'll assume the environment has a way to create these buttons
    
    running = True
    mouse_pos = (0, 0)
    observation = env.reset()

    # Initialize DQN agent
    model_path = "dqn_agent_best.pth"
    agent_ai = DQNAgent(19, 5)
    agent_ai.epsilon = 0
    agent_ai.policy_net.load_state_dict(torch.load(model_path, weights_only=True))
    agent_ai.policy_net.eval()
    print(f"Loaded DQN model from {model_path}")
    env.game_state = GameState.MENU
    # Main game loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p and env.game_state == GameState.PLAYING:
                    env.game_state = GameState.PAUSED
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
            # Select action based on selected mode
            if env.mode == "manual":
                # Manual play - human controls with arrow keys
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
            elif env.mode == "ai":
                # DQN agent plays autonomously
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = agent_ai.policy_net(state_tensor)
                action = q_values.argmax().item()
            elif env.mode == "coop":
                # Cooperative mode - human and DQN working together
                # Get DQN suggestion
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = agent_ai.policy_net(state_tensor)
                dqn_action = q_values.argmax().item()
                
                # Get human input
                keys = pygame.key.get_pressed()
                human_action = 0
                if keys[pygame.K_UP]:
                    human_action = 1
                elif keys[pygame.K_DOWN]:
                    human_action = 2
                elif keys[pygame.K_LEFT]:
                    human_action = 3
                elif keys[pygame.K_RIGHT]:
                    human_action = 4
                
                # If human provides input, use it; otherwise use DQN's suggestion
                action = human_action if human_action != 0 else dqn_action

            observation, reward, done, info = env.step(action)
            if done:
                env.game_state = GameState.GAME_OVER

        env.render(env.mode)
    
    # When exiting game, close the environment
    env.close()


if __name__ == "__main__":
    main()