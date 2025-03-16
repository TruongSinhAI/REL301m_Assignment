import math
import numpy as np
import pygame
import random
from enum import Enum
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Giả sử các module bên dưới đã được định nghĩa trong src:
from src.agents.frog import BaseFrog, Frog
from src.agents.vehicle import BaseVehicle, Vehicle
from src.constants.config import GAME_CONFIG, COLORS
from src.utils.button import Button

class GameState(Enum):
    MENU = 1
    PLAYING = 2
    PAUSED = 3
    GAME_OVER = 4

class BaseCarCrossingEnv:
    def __init__(self):
        # Thiết lập các thông số đường và làn xe
        self.total_lanes = GAME_CONFIG['TOTAL_LANES']
        self.lane_height = GAME_CONFIG['LANE_HEIGHT']
        self.road_top = GAME_CONFIG['ROAD_TOP']
        self.road_bottom = self.road_top + self.total_lanes * self.lane_height + 20

        # Cấu hình spawn xe
        self.min_vehicle_spacing = GAME_CONFIG['MIN_VEHICLE_SPACING']
        self.min_vehicle_width = GAME_CONFIG['MIN_VEHICLE_WIDTH']
        self.max_vehicle_width = GAME_CONFIG['MAX_VEHICLE_WIDTH']
        self.min_vehicle_height = GAME_CONFIG['MIN_VEHICLE_HEIGHT']
        self.max_vehicle_height = GAME_CONFIG['MAX_VEHICLE_HEIGHT']
        self.min_vehicle_speed = GAME_CONFIG['MIN_VEHICLE_SPEED']
        self.max_vehicle_speed = GAME_CONFIG['MAX_VEHICLE_SPEED']

        # Số lượng agent (mặc định 2, có thể cấu hình lại qua GAME_CONFIG)
        self.num_agents = GAME_CONFIG.get('NUM_AGENTS', 10)
        self.agents = [BaseFrog() for _ in range(self.num_agents)]
        self._init_agents_positions()

        self.vehicles = []
        self.spawn_vehicles()

        # Khởi tạo trạng thái cho từng agent
        self.done_flags = [False] * self.num_agents
        self.action_space = [0, 1, 2, 3, 4]  # 5 action: 0 - không di chuyển, 1 - lên, 2 - xuống, 3 - trái, 4 - phải
        self.scores = [0] * self.num_agents
        self.current_lanes = [self.get_lane_index(agent.y) for agent in self.agents]
        self.game_state = GameState.PLAYING
        self.mode = "ai"  # Mặc định chế độ AI

    def _init_agents_positions(self):
        """Khởi tạo vị trí cho các agent, chia đều trên trục x."""
        screen_width = GAME_CONFIG['SCREEN_WIDTH']
        for idx, agent in enumerate(self.agents):
            agent.y = self.road_bottom - agent.height - 10
            spacing = screen_width // (self.num_agents + 1)
            agent.x = spacing * (idx + 1) - agent.width // 2

    def spawn_vehicles(self):
        """Tạo các xe trong môi trường theo từng lane."""
        self.vehicles = []
        lane_width = GAME_CONFIG['SCREEN_WIDTH'] + 300
        for lane_index in range(self.total_lanes):
            # Bỏ qua lane nghỉ (mỗi lane thứ 3)
            if lane_index % 3 == 2:
                continue
            lane_y = self.road_top + 20 + lane_index * self.lane_height + 10
            direction = 1 if lane_index % 2 == 0 else -1
            available_space = lane_width
            vehicles_in_lane = []
            current_x = 0

            while available_space > GAME_CONFIG['MIN_VEHICLE_SPACING']:
                width = 40
                height = 40
                speed = random.randint(self.min_vehicle_speed, self.max_vehicle_speed) + random.random()
                # Tạo x_position ngẫu nhiên để tạo khoảng cách giữa các xe
                x_position = current_x + random.randint(0, 200)
                vehicle = BaseVehicle(x_position, lane_y, speed, direction, width, height, lane_index)
                vehicles_in_lane.append(vehicle)
                vehicle_spacing = width + GAME_CONFIG['MIN_VEHICLE_SPACING']
                available_space -= vehicle_spacing
                current_x += vehicle_spacing

            self.vehicles.extend(vehicles_in_lane)

    def get_lane_index(self, y_pos):
        lane = int((y_pos - (self.road_top + 20)) // self.lane_height)
        return max(lane, 0)

    def step(self, actions):
        """
        Hàm step nhận vào danh sách các action, mỗi action tương ứng với một agent.
        Trả về:
            observations: danh sách quan sát của các agent
            rewards: danh sách phần thưởng của các agent
            done: flag kết thúc (True nếu tất cả agent đều dừng)
            info: thông tin bổ sung (ví dụ: điểm số)
        """
        rewards = [0] * self.num_agents
        observations = []
        action_deltas = {
            0: (0, 0),
            1: (0, -1),  # Up
            2: (0, 1),   # Down
            3: (-1, 0),  # Left
            4: (1, 0)    # Right
        }

        # Cập nhật trạng thái cho từng agent
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            if self.done_flags[i]:
                observations.append(self.get_observation_for_agent(agent))
                continue

            prev_y = agent.y
            dx, dy = action_deltas.get(action, (0, 0))
            agent.move(dx, dy)

            # Giới hạn vị trí của agent trong khu vực cho phép
            agent.y = max(self.road_top, min(agent.y, self.road_bottom - agent.height))
            agent.x = max(0, min(agent.x, GAME_CONFIG['SCREEN_WIDTH'] - agent.width))

            # TÍNH PHẦN THƯỞNG cho agent
            progress_reward = (prev_y - agent.y) * 2.0
            distance_to_finish = agent.y - self.road_top
            normalized_distance = distance_to_finish / (self.road_bottom - self.road_top)
            finish_line_bonus = (1 - normalized_distance) * 10

            # Phạt khi quá gần xe: tính khoảng cách giữa agent và xe gần nhất
            agent_center = np.array([agent.x + agent.width / 2, agent.y + agent.height / 2])
            min_distance = float('inf')
            for vehicle in self.vehicles:
                vehicle_center = np.array([vehicle.x + vehicle.width / 2, vehicle.y + vehicle.height / 2])
                distance = np.linalg.norm(vehicle_center - agent_center)
                min_distance = min(min_distance, distance)
            safe_distance = 50.0
            proximity_penalty = -10 * np.exp(-min_distance / safe_distance)

            # Phạt thời gian
            time_penalty = -0.2

            # Thưởng chuyển sang lane cao hơn
            lane_bonus = 0
            new_lane = self.get_lane_index(agent.y)
            if new_lane < self.current_lanes[i]:
                lane_bonus = (self.current_lanes[i] - new_lane) * 5
                self.current_lanes[i] = new_lane
                self.scores[i] += lane_bonus

            reward = progress_reward + finish_line_bonus + proximity_penalty + time_penalty + lane_bonus
            rewards[i] = reward

        # Cập nhật vị trí xe
        for vehicle in self.vehicles:
            vehicle.update(self.vehicles)

        # Kiểm tra va chạm và trạng thái thắng/thua cho từng agent
        for i, agent in enumerate(self.agents):
            if self.done_flags[i]:
                observations.append(self.get_observation_for_agent(agent))
                continue
            for vehicle in self.vehicles:
                if self._check_collision(agent, vehicle):
                    rewards[i] = -100
                    self.done_flags[i] = True
                    break
            if agent.y <= self.road_top:
                rewards[i] = 500
                self.scores[i] += 100
                self.done_flags[i] = True

            observations.append(self.get_observation_for_agent(agent))

        overall_done = all(self.done_flags)
        info = {"scores": self.scores}
        return observations, rewards, overall_done, info

    def _check_collision(self, agent, vehicle):
        return (agent.x < vehicle.x + vehicle.width and
                agent.x + agent.width > vehicle.x and
                agent.y < vehicle.y + vehicle.height and
                agent.y + agent.height > vehicle.y)

    def get_observation_for_agent(self, agent):
        """Trả về vector quan sát cho agent dựa trên vị trí của nó và các xe xung quanh."""
        agent_x_norm = agent.x / GAME_CONFIG['SCREEN_WIDTH']
        agent_y_norm = agent.y / GAME_CONFIG['SCREEN_HEIGHT']
        lane_norm = self.get_lane_index(agent.y) / self.total_lanes
        risk_norm = (self._calculate_collision_risk_for_agent(agent) + 1)
        agent_center = np.array([agent.x + agent.width / 2, agent.y + agent.height / 2])
        vehicles_features = []
        for vehicle in self.vehicles:
            vehicle_center = np.array([vehicle.x + vehicle.width / 2, vehicle.y + vehicle.height / 2])
            rel_pos = (vehicle_center - agent_center) / np.array([GAME_CONFIG['SCREEN_WIDTH'], GAME_CONFIG['SCREEN_HEIGHT']])
            speed_norm = (vehicle.speed - self.min_vehicle_speed) / (self.max_vehicle_speed - self.min_vehicle_speed)
            vehicles_features.append(np.concatenate([rel_pos, [speed_norm]]))
        vehicles_features.sort(key=lambda feat: np.linalg.norm(feat[:2]))
        N = 5  # Lấy thông tin 5 xe gần nhất
        obs_vehicles = np.zeros(3 * N)
        for i in range(min(len(vehicles_features), N)):
            obs_vehicles[i * 3:(i + 1) * 3] = vehicles_features[i]
        observation = np.concatenate([
            np.array([agent_x_norm, agent_y_norm, lane_norm, risk_norm]),
            obs_vehicles
        ])
        return observation.astype(np.float32)

    def _calculate_collision_risk_for_agent(self, agent):
        min_distance = float('inf')
        agent_center = (agent.x + agent.width / 2, agent.y + agent.height / 2)
        for vehicle in self.vehicles:
            vehicle_center = (vehicle.x + vehicle.width / 2, vehicle.y + vehicle.height / 2)
            distance = math.sqrt((agent_center[0] - vehicle_center[0]) ** 2 +
                                 (agent_center[1] - vehicle_center[1]) ** 2)
            min_distance = min(min_distance, distance)
        safe_distance = 50.0
        risk = -1 + (min(min_distance, safe_distance) / safe_distance)
        return risk

    def reset(self):
        """Reset môi trường: khởi tạo lại các agent và xe."""
        self.agents = [BaseFrog() for _ in range(self.num_agents)]
        self._init_agents_positions()
        self.spawn_vehicles()
        self.done_flags = [False] * self.num_agents
        self.scores = [0] * self.num_agents
        self.current_lanes = [self.get_lane_index(agent.y) for agent in self.agents]
        self.game_state = GameState.PLAYING
        observations = [self.get_observation_for_agent(agent) for agent in self.agents]
        return observations

    def close(self):
        pygame.quit()


# Môi trường có giao diện GUI cho multi-agent (sử dụng Frog thay cho BaseFrog)
class MultiAgentCarCrossingEnvGUI(BaseCarCrossingEnv):
    def __init__(self, num_agents=2):
        pygame.init()
        pygame.mixer.init()
        self.num_agents = num_agents
        super().__init__()
        # Ghi đè agent bằng phiên bản có giao diện
        self.agents = [Frog() for _ in range(self.num_agents)]
        self._init_agents_positions()
        self.vehicles = []
        self.spawn_vehicles()

        self.screen = pygame.display.set_mode((GAME_CONFIG['SCREEN_WIDTH'], GAME_CONFIG['SCREEN_HEIGHT']))
        pygame.display.set_caption("Multi-Agent Car Crossing Game")
        self.clock = pygame.time.Clock()

        # Khởi tạo font và buttons cho GUI
        self.FPS_font = pygame.font.SysFont("Helvetica", 24)
        self.title_font = pygame.font.SysFont("Helvetica", 72, bold=True)
        self.menu_font = pygame.font.SysFont("Helvetica", 36)
        self.game_over_font = pygame.font.SysFont("Helvetica", 64, bold=True)
        self.score_font = pygame.font.SysFont("Helvetica", 32)
        self.instruction_font = pygame.font.SysFont("Helvetica", 24)
        self._setup_buttons()

        self.collision_sound = None
        self.win_sound = None

    def _setup_buttons(self):
        button_width = 200
        button_height = 50
        menu_button_y_start = GAME_CONFIG['SCREEN_HEIGHT'] // 2 - (button_height * 2) // 2
        button_x = GAME_CONFIG['SCREEN_WIDTH'] // 2 - button_width // 2

        self.manual_button = Button(
            "Manual Mode",
            button_x,
            menu_button_y_start,
            button_width,
            button_height,
            COLORS['BUTTON_GRAY'],
            COLORS['BUTTON_HOVER_GRAY'],
            self.menu_font,
            action=lambda: self.set_mode("manual")
        )
        self.ai_button = Button(
            "AI Mode",
            button_x,
            menu_button_y_start + button_height + 20,
            button_width,
            button_height,
            COLORS['BUTTON_GRAY'],
            COLORS['BUTTON_HOVER_GRAY'],
            self.menu_font,
            action=lambda: self.set_mode("ai")
        )
        self.menu_buttons = [self.manual_button, self.ai_button]

        game_over_button_y_start = GAME_CONFIG['SCREEN_HEIGHT'] // 2 + 50
        self.restart_button = Button(
            "Restart",
            button_x,
            game_over_button_y_start,
            button_width,
            button_height,
            COLORS['BUTTON_GRAY'],
            COLORS['BUTTON_HOVER_GRAY'],
            self.instruction_font,
            action=lambda: self.restart_game()
        )
        self.menu_return_button = Button(
            "Menu",
            button_x,
            game_over_button_y_start + button_height + 10,
            button_width,
            button_height,
            COLORS['BUTTON_GRAY'],
            COLORS['BUTTON_HOVER_GRAY'],
            self.instruction_font,
            action=lambda: self.return_to_menu()
        )
        self.game_over_buttons = [self.restart_button, self.menu_return_button]

    def spawn_vehicles(self):
        self.vehicles = []
        lane_width = GAME_CONFIG['SCREEN_WIDTH'] + 300
        for lane_index in range(self.total_lanes):
            if lane_index % 3 == 2:
                continue
            lane_y = self.road_top + 20 + lane_index * self.lane_height + 10
            direction = 1 if lane_index % 2 == 0 else -1
            available_space = lane_width
            vehicles_in_lane = []
            current_x = 0

            while available_space > GAME_CONFIG['MIN_VEHICLE_SPACING']:
                width = random.randint(self.min_vehicle_width, self.max_vehicle_width)
                height = random.randint(self.min_vehicle_height, self.max_vehicle_height)
                speed = random.randint(self.min_vehicle_speed, self.max_vehicle_speed) + random.random()
                x_position = current_x + random.randint(0, 200)
                vehicle = Vehicle(x_position, lane_y, speed, direction, width, height, lane_index)
                vehicles_in_lane.append(vehicle)
                vehicle_spacing = width + GAME_CONFIG['MIN_VEHICLE_SPACING']
                available_space -= vehicle_spacing
                current_x += vehicle_spacing

            self.vehicles.extend(vehicles_in_lane)

    def step(self, actions):
        observations, rewards, done, info = super().step(actions)
        # Phát âm thanh nếu cần (va chạm hoặc chiến thắng)
        for reward in rewards:
            if reward < 0:
                try:
                    self.collision_sound.play()
                except:
                    pass
            elif reward > 0:
                try:
                    self.win_sound.play()
                except:
                    pass
        return observations, rewards, done, info

    def render(self, mode="manual"):
        if self.game_state == GameState.MENU:
            self.render_menu()
            return
        elif self.game_state == GameState.GAME_OVER:
            self.render_game_over()
            return
        elif self.game_state == GameState.PAUSED:
            self.render_pause_screen()
            return

        # Sử dụng vị trí của agent đầu tiên để tính camera_offset (có thể cải tiến cho nhiều agent)
        camera_offset = max(0, min(self.agents[0].y - (GAME_CONFIG['SCREEN_HEIGHT'] - 150),
                                     self.road_bottom - GAME_CONFIG['SCREEN_HEIGHT']))
        self.screen.fill(COLORS['GRASS'])
        self._draw_lanes(camera_offset)

        finish_line_y = self.road_top - camera_offset
        pygame.draw.line(self.screen, COLORS['GOLD'], (0, finish_line_y),
                         (GAME_CONFIG['SCREEN_WIDTH'], finish_line_y), 5)

        for agent in self.agents:
            agent.draw(self.screen, camera_offset)
        for vehicle in self.vehicles:
            vehicle.draw(self.screen, camera_offset)

        self._draw_game_info(mode)
        pygame.display.flip()
        self.clock.tick(GAME_CONFIG['FPS'])

    def _draw_lanes(self, camera_offset):
        vehicle_lane_color = COLORS['VEHICLE_LANE']
        rest_lane_color = COLORS['REST_LANE']
        dash_color = COLORS['LIGHT_GRAY']
        dash_length = GAME_CONFIG['DASH_LENGTH']
        lane_height = GAME_CONFIG['LANE_HEIGHT']

        for lane_index in range(self.total_lanes):
            lane_y = self.road_top + 20 + lane_index * lane_height - camera_offset
            lane_rect = pygame.Rect(0, lane_y, GAME_CONFIG['SCREEN_WIDTH'], lane_height)
            if lane_index % 3 == 2:
                pygame.draw.rect(self.screen, rest_lane_color, lane_rect)
            else:
                pygame.draw.rect(self.screen, vehicle_lane_color, lane_rect)
                for x in range(0, GAME_CONFIG['SCREEN_WIDTH'], 2 * dash_length):
                    pygame.draw.line(self.screen, dash_color,
                                     (x, lane_y + lane_height // 2),
                                     (x + dash_length, lane_y + lane_height // 2), 3)

    def _draw_game_info(self, mode):
        FPS = self.clock.get_fps()
        FPS_text = self.FPS_font.render(f"FPS: {int(FPS)}", True, COLORS['WHITE'])
        self.screen.blit(FPS_text, (10, 40))
        mode_text = self.score_font.render(f"Mode: {mode.upper()}", True, COLORS['GOLD'])
        score_text = self.score_font.render(f"Scores: {self.scores}", True, COLORS['GOLD'])
        self.screen.blit(mode_text, (10, 10))
        self.screen.blit(score_text, (GAME_CONFIG['SCREEN_WIDTH'] - score_text.get_width() - 10, 10))

    def render_menu(self):
        self.screen.fill(COLORS['GRASS'])
        title_text = self.title_font.render("Multi-Agent Car Crossing Game", True, COLORS['WHITE'])
        title_rect = title_text.get_rect(center=(GAME_CONFIG['SCREEN_WIDTH'] // 2, GAME_CONFIG['SCREEN_HEIGHT'] // 3))
        self.screen.blit(title_text, title_rect)
        for button in self.menu_buttons:
            button.draw(self.screen)
        pygame.display.flip()

    def render_game_over(self):
        camera_offset = max(0, min(self.agents[0].y - (GAME_CONFIG['SCREEN_HEIGHT'] - 150),
                                     self.road_bottom - GAME_CONFIG['SCREEN_HEIGHT']))
        self.screen.fill(COLORS['GRASS'])
        self._draw_lanes(camera_offset)
        finish_line_y = self.road_top - camera_offset
        pygame.draw.line(self.screen, COLORS['GOLD'], (0, finish_line_y),
                         (GAME_CONFIG['SCREEN_WIDTH'], finish_line_y), 5)
        for agent in self.agents:
            agent.draw(self.screen, camera_offset)
        for vehicle in self.vehicles:
            vehicle.draw(self.screen, camera_offset)
        self._draw_game_info(self.mode)
        overlay = pygame.Surface((GAME_CONFIG['SCREEN_WIDTH'], GAME_CONFIG['SCREEN_HEIGHT']))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(128)
        self.screen.blit(overlay, (0, 0))
        game_over_text = self.game_over_font.render("Game Over!", True, COLORS['WHITE'])
        score_text = self.score_font.render(f"Scores: {self.scores}", True, COLORS['GOLD'])
        game_over_rect = game_over_text.get_rect(center=(GAME_CONFIG['SCREEN_WIDTH'] // 2, GAME_CONFIG['SCREEN_HEIGHT'] // 3))
        score_rect = score_text.get_rect(center=(GAME_CONFIG['SCREEN_WIDTH'] // 2, GAME_CONFIG['SCREEN_HEIGHT'] // 2 - 30))
        self.screen.blit(game_over_text, game_over_rect)
        self.screen.blit(score_text, score_rect)
        for button in self.game_over_buttons:
            button.draw(self.screen)
        pygame.display.flip()

    def render_pause_screen(self):
        overlay = pygame.Surface((GAME_CONFIG['SCREEN_WIDTH'], GAME_CONFIG['SCREEN_HEIGHT']))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(128)
        self.screen.blit(overlay, (0, 0))
        pause_text = self.game_over_font.render("PAUSED", True, COLORS['WHITE'])
        continue_text = self.instruction_font.render("Press P to Continue", True, COLORS['WHITE'])
        pause_rect = pause_text.get_rect(center=(GAME_CONFIG['SCREEN_WIDTH'] // 2, GAME_CONFIG['SCREEN_HEIGHT'] // 3))
        continue_rect = continue_text.get_rect(center=(GAME_CONFIG['SCREEN_WIDTH'] // 2, GAME_CONFIG['SCREEN_HEIGHT'] // 2))
        self.screen.blit(pause_text, pause_rect)
        self.screen.blit(continue_text, continue_rect)
        pygame.display.flip()

    def set_mode(self, mode_str):
        self.mode = mode_str
        self.game_state = GameState.PLAYING

    def restart_game(self):
        self.reset()
        self.game_state = GameState.PLAYING

    def return_to_menu(self):
        self.game_state = GameState.MENU
        self.mode = "menu"

    def reset(self):
        self.agents = [Frog() for _ in range(self.num_agents)]
        self._init_agents_positions()
        self.spawn_vehicles()
        self.done_flags = [False] * self.num_agents
        self.scores = [0] * self.num_agents
        self.current_lanes = [self.get_lane_index(agent.y) for agent in self.agents]
        self.game_state = GameState.PLAYING
        observations = [self.get_observation_for_agent(agent) for agent in self.agents]
        return observations

    def close(self):
        pygame.quit()
