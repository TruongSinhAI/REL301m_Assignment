import math
import numpy as np
import torch
import pygame
import random
from enum import Enum

from src.agents.frog import *
from src.agents.vehicle import *
from src.constants.config import *
from src.utils.button import Button


# Giả sử GAME_CONFIG và COLORS đã được định nghĩa ở nơi khác,
# cũng như các lớp BaseFrog, Frog, BaseVehicle, Vehicle, Button

class GameState(Enum):
    MENU = 1
    PLAYING = 2
    PAUSED = 3
    GAME_OVER = 4

class BaseCarCrossingEnv:
    def __init__(self):
        # Road and lane setup
        self.total_lanes = GAME_CONFIG['TOTAL_LANES']
        self.lane_height = GAME_CONFIG['LANE_HEIGHT']
        self.road_top = GAME_CONFIG['ROAD_TOP']
        self.road_bottom = self.road_top + self.total_lanes * self.lane_height + 20

        # Vehicle spawn configuration
        self.min_vehicle_spacing = GAME_CONFIG['MIN_VEHICLE_SPACING']
        self.min_vehicle_width = GAME_CONFIG['MIN_VEHICLE_WIDTH']
        self.max_vehicle_width = GAME_CONFIG['MAX_VEHICLE_WIDTH']
        self.min_vehicle_height = GAME_CONFIG['MIN_VEHICLE_HEIGHT']
        self.max_vehicle_height = GAME_CONFIG['MAX_VEHICLE_HEIGHT']
        self.min_vehicle_speed = GAME_CONFIG['MIN_VEHICLE_SPEED']
        self.max_vehicle_speed = GAME_CONFIG['MAX_VEHICLE_SPEED']

        self.agent = BaseFrog()
        self.agent.y = self.road_bottom - self.agent.height - 10
        self.agent.x = GAME_CONFIG['SCREEN_WIDTH'] // 2 - self.agent.width // 2

        self.vehicles = []
        self.spawn_vehicles()

        self.done = False
        self.action_space = [0, 1, 2, 3, 4]
        self.score = 0
        self.current_lane = self.get_lane_index(self.agent.y)
        self.game_state = GameState.PLAYING
        self.mode = "ai"  # Default to AI mode

    def spawn_vehicles(self):
        self.vehicles = []
        lane_width = GAME_CONFIG['SCREEN_WIDTH'] + 300
        for lane_index in range(self.total_lanes):
            if lane_index % 3 == 2:  # Skip rest lanes
                continue
            lane_y = self.road_top + 20 + lane_index * self.lane_height
            direction = 1 if lane_index % 2 == 0 else -1
            available_space = lane_width
            vehicles_in_lane = []
            current_x = 0

            while available_space > GAME_CONFIG['MIN_VEHICLE_SPACING']:
                # Có thể dùng kích thước xe cố định hoặc ngẫu nhiên
                width = 40
                height = 40
                speed = random.randint(self.min_vehicle_speed, self.max_vehicle_speed) + random.random()
                x_position = current_x
                if direction == -1:
                    x_position = GAME_CONFIG['SCREEN_WIDTH'] + current_x
                vehicle = BaseVehicle(x_position, lane_y, speed, direction, width, height, lane_index)
                vehicles_in_lane.append(vehicle)
                vehicle_spacing = width + GAME_CONFIG['MIN_VEHICLE_SPACING']
                available_space -= vehicle_spacing
                current_x += vehicle_spacing

            self.vehicles.extend(vehicles_in_lane)

    def get_lane_index(self, y_pos):
        lane = int((y_pos - (self.road_top + 20)) // self.lane_height)
        return max(lane, 0)

    # --- HÀM STEP ĐÃ ĐƯỢC CẢI TIẾN ---
    def step(self, action):
        # Lưu trạng thái ban đầu
        prev_x, prev_y = self.agent.x, self.agent.y

        # Cập nhật điểm cao nhất nếu cần
        try:
            self.highest = min(self.highest, self.agent.y)
        except:
            self.highest = self.agent.y

        # Định nghĩa 5 action: 0: không di chuyển, 1: lên, 2: xuống, 3: trái, 4: phải
        action_deltas = {
            0: (0, 0),
            1: (0, -1),  # Up
            2: (0, 1),   # Down
            3: (-1, 0),  # Left
            4: (1, 0)    # Right
        }

        # Thực hiện action đã chọn
        dx, dy = action_deltas.get(action, (0, 0))
        self.agent.move(dx, dy)

        # Giới hạn vị trí agent trong vùng cho phép
        self.agent.y = max(self.road_top, min(self.agent.y, self.road_bottom - self.agent.height))
        self.agent.x = max(0, min(self.agent.x, GAME_CONFIG['SCREEN_WIDTH'] - self.agent.width))

        # Cập nhật vị trí của các xe
        for vehicle in self.vehicles:
            vehicle.update(self.vehicles)

        # Lưu lại trạng thái xe sau cập nhật để dùng cho mô phỏng
        current_vehicles = [(v.x, v.y, v.width, v.height) for v in self.vehicles]

        # Tính phần thưởng tiến lên
        vertical_progress = prev_y - self.agent.y
        progress_reward = vertical_progress * 2.0

        # Mô phỏng các action thay thế từ trạng thái ban đầu
        alternative_bonus = 0
        reward_per_loss = 10  # Tăng bonus để khuyến khích chọn action an toàn

        def rect_collision(rect1, rect2):
            x1, y1, w1, h1 = rect1
            x2, y2, w2, h2 = rect2
            return not (x1 + w1 <= x2 or x1 >= x2 + w2 or y1 + h1 <= y2 or y1 >= y2 + h2)

        for alt_action, (dx_alt, dy_alt) in action_deltas.items():
            if alt_action == action:
                continue  # Bỏ qua action đã chọn

            sim_x = prev_x + dx_alt
            sim_y = prev_y + dy_alt

            # Giới hạn vị trí giả lập
            sim_y = max(self.road_top, min(sim_y, self.road_bottom - self.agent.height))
            sim_x = max(0, min(sim_x, GAME_CONFIG['SCREEN_WIDTH'] - self.agent.width))

            sim_rect = (sim_x, sim_y, self.agent.width, self.agent.height)

            collision = False
            for (vx, vy, vwidth, vheight) in current_vehicles:
                if rect_collision(sim_rect, (vx, vy, vwidth, vheight)):
                    collision = True
                    break
            if collision:
                alternative_bonus += reward_per_loss

        # Thêm “safety penalty” dựa trên khoảng cách đến xe gần nhất
        safety_penalty = self._calculate_collision_risk() * 5  # Hệ số điều chỉnh

        # Thêm time penalty để khuyến khích hoàn thành nhanh
        time_penalty = -0.2

        # Tổng hợp reward
        reward = progress_reward + alternative_bonus + safety_penalty + time_penalty

        # Kiểm tra va chạm thực tế
        for vehicle in self.vehicles:
            if self._check_collision(self.agent, vehicle):
                self.done = True
                reward = -100  # Phạt va chạm
                self.game_state = GameState.GAME_OVER
                break

        # Kiểm tra điều kiện thắng (agent đạt tới đích)
        if self.agent.y <= self.road_top:
            self.done = True
            reward = 100  # Phần thưởng thắng
            self.score += 100
            self.game_state = GameState.GAME_OVER

        # Cập nhật score theo làn agent đã vượt qua
        self.update_score()

        observation = self.get_observation()
        return observation, reward, self.done, self.score

    def _calculate_collision_risk(self):
        """Tính toán mức độ nguy hiểm dựa trên khoảng cách đến xe gần nhất."""
        min_distance = float('inf')
        for vehicle in self.vehicles:
            dx = max(0, max(self.agent.x - (vehicle.x + vehicle.width), vehicle.x - (self.agent.x + self.agent.width)))
            dy = max(0, max(self.agent.y - (vehicle.y + vehicle.height), vehicle.y - (self.agent.y + self.agent.height)))
            distance = math.sqrt(dx ** 2 + dy ** 2)
            min_distance = min(min_distance, distance)
        # Ánh xạ khoảng cách sang risk score: càng gần (distance thấp) thì risk càng âm
        if min_distance < 1:
            return -1.0
        elif min_distance < 3:
            return -0.5
        elif min_distance < 5:
            return -0.2
        else:
            return 0

    def _check_collision(self, agent, vehicle):
        return (agent.x < vehicle.x + vehicle.width and
                agent.x + agent.width > vehicle.x and
                agent.y < vehicle.y + vehicle.height and
                agent.y + agent.height > vehicle.y)

    def update_score(self):
        new_lane = self.get_lane_index(self.agent.y)
        if new_lane < self.current_lane:
            self.score += (self.current_lane - new_lane) * 5
            self.current_lane = new_lane

    # --- CẢI TIẾN GET_OBSERVATION: THÊM EXTRA CHANNEL VỚI RISK ---
    def get_observation(self):
        cell_size = 40
        grid_width = GAME_CONFIG['SCREEN_WIDTH'] // cell_size
        grid_height = GAME_CONFIG['SCREEN_HEIGHT'] // cell_size

        # Khởi tạo grid cho agent và xe
        grid = np.zeros((grid_height, grid_width), dtype=np.float32)

        # Đánh dấu vị trí của agent với giá trị 1
        agent_grid_x = min(max(0, int(self.agent.x // cell_size)), grid_width - 1)
        agent_grid_y = min(max(0, int(self.agent.y // cell_size)), grid_height - 1)
        grid[agent_grid_y][agent_grid_x] = 1.0

        # Đánh dấu xe với giá trị 2
        for vehicle in self.vehicles:
            min_x = max(0, int(vehicle.x // cell_size))
            max_x = min(grid_width - 1, int((vehicle.x + vehicle.width) // cell_size))
            min_y = max(0, int(vehicle.y // cell_size))
            max_y = min(grid_height - 1, int((vehicle.y + vehicle.height) // cell_size))
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    grid[y][x] = 2.0

        # Chuẩn hóa grid về [0, 1]
        grid = grid / 2.0

        # Tạo extra channel với risk: sử dụng _calculate_collision_risk
        risk = self._calculate_collision_risk()  # Giá trị trong khoảng [-1, 0]
        risk_channel = np.full((grid_height, grid_width), risk, dtype=np.float32)
        # Chuẩn hóa risk_channel: -1 -> 0 (nguy hiểm) và 0 -> 1 (an toàn)
        risk_channel = (risk_channel + 1) / 1.0

        # Xếp chồng 2 kênh để tạo observation (shape: [2, grid_height, grid_width])
        observation = np.stack([grid, risk_channel], axis=0)
        return observation

    def reset(self):
        self.agent = BaseFrog()
        self.agent.x = GAME_CONFIG['SCREEN_WIDTH'] // 2 - self.agent.width // 2
        self.agent.y = self.road_bottom - self.agent.height - 10
        self.spawn_vehicles()
        self.done = False
        self.score = 0
        self.current_lane = self.get_lane_index(self.agent.y)
        self.game_state = GameState.PLAYING
        return self.get_observation()

    def close(self):
        pygame.quit()

# --- PHẦN ENVIRONMENT CHO GIAO DIỆN (GUI) ---
class CarCrossingEnv(BaseCarCrossingEnv):
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        super().__init__()

        self.screen = pygame.display.set_mode((GAME_CONFIG['SCREEN_WIDTH'], GAME_CONFIG['SCREEN_HEIGHT']))
        pygame.display.set_caption("Car Crossing Game")
        self.clock = pygame.time.Clock()

        # Override agent và vehicles với phiên bản GUI
        self.agent = Frog()
        self.agent.y = self.road_bottom - self.agent.height - 10
        self.agent.x = GAME_CONFIG['SCREEN_WIDTH'] // 2 - self.agent.width // 2
        self.vehicles = []
        self.spawn_vehicles()

        # Fonts và Buttons
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
            lane_y = self.road_top + 20 + lane_index * self.lane_height
            direction = 1 if lane_index % 2 == 0 else -1
            available_space = lane_width
            vehicles_in_lane = []
            current_x = 0

            while available_space > GAME_CONFIG['MIN_VEHICLE_SPACING']:
                width = random.randint(self.min_vehicle_width, self.max_vehicle_width)
                height = random.randint(self.min_vehicle_height, self.max_vehicle_height)
                speed = random.randint(self.min_vehicle_speed, self.max_vehicle_speed) + random.random()
                x_position = current_x
                if direction == -1:
                    x_position = GAME_CONFIG['SCREEN_WIDTH'] + current_x
                vehicle = Vehicle(x_position, lane_y, speed, direction, width, height, lane_index)
                vehicles_in_lane.append(vehicle)
                vehicle_spacing = width + GAME_CONFIG['MIN_VEHICLE_SPACING']
                available_space -= vehicle_spacing
                current_x += vehicle_spacing

            self.vehicles.extend(vehicles_in_lane)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        if done:
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
        return observation, reward, done, info

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

        camera_offset = max(0, min(self.agent.y - (GAME_CONFIG['SCREEN_HEIGHT'] - 150),
                                   self.road_bottom - GAME_CONFIG['SCREEN_HEIGHT']))

        self.screen.fill(COLORS['GRASS'])
        self._draw_lanes(camera_offset)

        finish_line_y = self.road_top - camera_offset
        pygame.draw.line(self.screen, COLORS['GOLD'], (0, finish_line_y),
                         (GAME_CONFIG['SCREEN_WIDTH'], finish_line_y), 5)

        self.agent.draw(self.screen, camera_offset)
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
        score_text = self.score_font.render(f"Score: {self.score}", True, COLORS['GOLD'])
        self.screen.blit(mode_text, (10, 10))
        self.screen.blit(score_text, (GAME_CONFIG['SCREEN_WIDTH'] - score_text.get_width() - 10, 10))

    def render_menu(self):
        self.screen.fill(COLORS['GRASS'])
        title_text = self.title_font.render("Car Crossing Game", True, COLORS['WHITE'])
        title_rect = title_text.get_rect(center=(GAME_CONFIG['SCREEN_WIDTH'] // 2, GAME_CONFIG['SCREEN_HEIGHT'] // 3))
        self.screen.blit(title_text, title_rect)
        for button in self.menu_buttons:
            button.draw(self.screen)
        pygame.display.flip()

    def render_game_over(self):
        camera_offset = max(0, min(self.agent.y - (GAME_CONFIG['SCREEN_HEIGHT'] - 150),
                                   self.road_bottom - GAME_CONFIG['SCREEN_HEIGHT']))
        self.screen.fill(COLORS['GRASS'])
        self._draw_lanes(camera_offset)
        finish_line_y = self.road_top - camera_offset
        pygame.draw.line(self.screen, COLORS['GOLD'], (0, finish_line_y),
                         (GAME_CONFIG['SCREEN_WIDTH'], finish_line_y), 5)
        self.agent.draw(self.screen, camera_offset)
        for vehicle in self.vehicles:
            vehicle.draw(self.screen, camera_offset)
        self._draw_game_info(self.mode)
        overlay = pygame.Surface((GAME_CONFIG['SCREEN_WIDTH'], GAME_CONFIG['SCREEN_HEIGHT']))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(128)
        self.screen.blit(overlay, (0, 0))
        game_over_text = self.game_over_font.render("Game Over!", True, COLORS['WHITE'])
        score_text = self.score_font.render(f"Score: {self.score}", True, COLORS['GOLD'])
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
        self.agent = Frog()
        self.agent.x = GAME_CONFIG['SCREEN_WIDTH'] // 2 - self.agent.width // 2
        self.agent.y = self.road_bottom - self.agent.height - 10
        self.spawn_vehicles()
        self.done = False
        self.score = 0
        self.current_lane = self.get_lane_index(self.agent.y)
        self.game_state = GameState.PLAYING
        return self.get_observation()

    def close(self):
        pygame.quit()
