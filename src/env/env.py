import imp
from src.constants.config import *
from src.utils.resource_manager import ResourceManager
from src.agents.frog import *
from src.agents.vehicle import *
from src.utils.button import Button
import pygame
import random
from enum import Enum

class GameState(Enum):
    MENU = 1
    PLAYING = 2
    PAUSED = 3
    GAME_OVER = 4

class CarCrossingEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((GAME_CONFIG['SCREEN_WIDTH'], GAME_CONFIG['SCREEN_HEIGHT']))
        pygame.display.set_caption("Car Crossing Game")
        self.clock = pygame.time.Clock()
        self.FPS_font = pygame.font.SysFont("Helvetica", 24)
        self.game_state = GameState.MENU
        self.last_spawn_time = pygame.time.get_ticks()

        # Road and lane setup (giữ nguyên)
        self.total_lanes = GAME_CONFIG['TOTAL_LANES']
        self.lane_height = GAME_CONFIG['LANE_HEIGHT']
        self.road_top = GAME_CONFIG['ROAD_TOP']
        self.road_bottom = self.road_top + self.total_lanes * self.lane_height + 20

        # Vehicle spawn configuration (giữ nguyên)
        self.min_vehicle_spacing = GAME_CONFIG['MIN_VEHICLE_SPACING']
        self.min_vehicle_width = GAME_CONFIG['MIN_VEHICLE_WIDTH']
        self.max_vehicle_width = GAME_CONFIG['MAX_VEHICLE_WIDTH']
        self.min_vehicle_height = GAME_CONFIG['MIN_VEHICLE_HEIGHT']
        self.max_vehicle_height = GAME_CONFIG['MAX_VEHICLE_HEIGHT']
        self.min_vehicle_speed = GAME_CONFIG['MIN_VEHICLE_SPEED']
        self.max_vehicle_speed = GAME_CONFIG['MAX_VEHICLE_SPEED']

        self.agent = Frog()
        self.agent.y = self.road_bottom - self.agent.height - 10

        # Vehicle Pool removed, directly instantiate vehicles
        self.vehicles = []
        self.spawn_vehicles()

        self.done = False
        self.action_space = [0, 1, 2, 3, 4]
        self.score = 0
        self.current_lane = self.get_lane_index(self.agent.y)
        self.paused = False

        # Fonts (giữ nguyên)
        self.title_font = pygame.font.SysFont("Helvetica", 72, bold=True)
        self.menu_font = pygame.font.SysFont("Helvetica", 36)
        self.game_over_font = pygame.font.SysFont("Helvetica", 64, bold=True)
        self.score_font = pygame.font.SysFont("Helvetica", 32)
        self.instruction_font = pygame.font.SysFont("Helvetica", 24)

        # Menu Buttons (giữ nguyên hoặc dùng nút bấm hình ảnh nếu đã chỉnh sửa)
        button_width = 200
        button_height = 50
        menu_button_y_start = GAME_CONFIG['SCREEN_HEIGHT'] // 2 - (button_height * 2) // 2
        button_x = GAME_CONFIG['SCREEN_WIDTH'] // 2 - button_width // 2

        self.manual_button = Button("Manual Mode", button_x, menu_button_y_start, button_width, button_height, COLORS['BUTTON_GRAY'], COLORS['BUTTON_HOVER_GRAY'], self.menu_font, action=lambda: self.set_mode("manual"))
        self.ai_button = Button("AI Mode", button_x, menu_button_y_start + button_height + 20, button_width, button_height, COLORS['BUTTON_GRAY'], COLORS['BUTTON_HOVER_GRAY'], self.menu_font, action=lambda: self.set_mode("ai"))
        self.menu_buttons = [self.manual_button, self.ai_button]

        # Game Over Buttons (giữ nguyên hoặc dùng nút bấm hình ảnh nếu đã chỉnh sửa)
        game_over_button_y_start = GAME_CONFIG['SCREEN_HEIGHT'] // 2 + 50
        self.restart_button = Button("Restart", button_x, game_over_button_y_start, button_width, button_height, COLORS['BUTTON_GRAY'], COLORS['BUTTON_HOVER_GRAY'], self.instruction_font, action=lambda: self.restart_game())
        self.menu_return_button = Button("Menu", button_x, game_over_button_y_start + button_height + 10, button_width, button_height, COLORS['BUTTON_GRAY'], COLORS['BUTTON_HOVER_GRAY'], self.instruction_font, action=lambda: self.return_to_menu())
        self.game_over_buttons = [self.restart_button, self.menu_return_button]

        self.mode = "menu"

        # **Âm thanh**
        pygame.mixer.init() # Khởi tạo mixer (nếu chưa chắc chắn đã khởi tạo trước đó)
        self.resource_manager = ResourceManager()
        # Load sounds once
        self.collision_sound = self.resource_manager.get_sound("crash")
        self.win_sound = self.resource_manager.get_sound("win")


    def set_mode(self, mode_str):
        self.mode = mode_str
        self.game_state = GameState.PLAYING

    def restart_game(self):
        self.reset()
        self.game_state = GameState.PLAYING

    def return_to_menu(self):
        self.game_state = GameState.MENU
        self.mode = "menu"

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
                vehicle = Vehicle(x_position, lane_y, speed, direction, width, height, lane_index) # Use Vehicle class
                vehicles_in_lane.append(vehicle)
                vehicle_spacing = width + GAME_CONFIG['MIN_VEHICLE_SPACING']
                available_space -= vehicle_spacing
                current_x += vehicle_spacing
            self.vehicles.extend(vehicles_in_lane)

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

        camera_offset = max(0, min(self.agent.y - (GAME_CONFIG['SCREEN_HEIGHT'] - 150), self.road_bottom - GAME_CONFIG['SCREEN_HEIGHT']))

        self.screen.fill(COLORS['GRASS'])

        self._draw_lanes(camera_offset)

        finish_line_y = self.road_top - camera_offset
        pygame.draw.line(self.screen, COLORS['GOLD'], (0, finish_line_y), (GAME_CONFIG['SCREEN_WIDTH'], finish_line_y), 5)

        self.agent.draw(self.screen, camera_offset) # Use Frog.draw
        for vehicle in self.vehicles:
            vehicle.draw(self.screen, camera_offset) # Use Vehicle.draw

        self._draw_game_info(mode)

        pygame.display.flip()
        self.clock.tick(GAME_CONFIG['FPS'])

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
        # **Render game scene phía sau trước**
        camera_offset = max(0, min(self.agent.y - (GAME_CONFIG['SCREEN_HEIGHT'] - 150), self.road_bottom - GAME_CONFIG['SCREEN_HEIGHT']))
        self.screen.fill(COLORS['GRASS'])
        self._draw_lanes(camera_offset)
        finish_line_y = self.road_top - camera_offset
        pygame.draw.line(self.screen, COLORS['GOLD'], (0, finish_line_y), (GAME_CONFIG['SCREEN_WIDTH'], finish_line_y), 5)
        self.agent.draw(self.screen, camera_offset)
        for vehicle in self.vehicles:
            vehicle.draw(self.screen, camera_offset)
        self._draw_game_info(self.mode) # Cập nhật thông tin game (FPS, Score, Mode) trên nền game over

        # **Vẽ lớp overlay mờ lên trên**
        overlay = pygame.Surface((GAME_CONFIG['SCREEN_WIDTH'], GAME_CONFIG['SCREEN_HEIGHT']))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(128)
        self.screen.blit(overlay, (0, 0))

        # **Vẽ chữ "Game Over" và điểm số**
        game_over_text = self.game_over_font.render("Game Over!", True, COLORS['WHITE'])
        score_text = self.score_font.render(f"Score: {self.score}", True, COLORS['GOLD'])
        game_over_rect = game_over_text.get_rect(center=(GAME_CONFIG['SCREEN_WIDTH'] // 2, GAME_CONFIG['SCREEN_HEIGHT'] // 3))
        score_rect = score_text.get_rect(center=(GAME_CONFIG['SCREEN_WIDTH'] // 2, GAME_CONFIG['SCREEN_HEIGHT'] // 2 - 30))
        self.screen.blit(game_over_text, game_over_rect)
        self.screen.blit(score_text, score_rect)

        # **Vẽ các nút bấm Game Over**
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
                    pygame.draw.line(self.screen, dash_color, (x, lane_y + lane_height // 2), (x + dash_length, lane_y + lane_height // 2), 3)


    def get_lane_index(self, y_pos):
        lane = int((y_pos - (self.road_top + 20)) // self.lane_height)
        return max(lane, 0)

    def step(self, action):
        if action == 1:
            self.agent.move(0, -1)
        elif action == 2:
            self.agent.move(0, 1)
        elif action == 3:
            self.agent.move(-1, 0)
        elif action == 4:
            self.agent.move(1, 0)

        self.agent.y = max(self.road_top, min(self.agent.y, self.road_bottom - self.agent.height))

        for vehicle in self.vehicles:
            vehicle.update(self.vehicles) # Use Vehicle.update

        self.update_score()

        reward = 0
        agent_rect = self.agent.get_rect() # Use Frog.get_rect

        for vehicle in self.vehicles:
            if agent_rect.colliderect(vehicle.get_rect()): # Use Vehicle.get_rect
                self.done = True
                reward = -10
                self.collision_sound.play()
                break

        if self.agent.y <= self.road_top:
            self.done = True
            reward = 10
            self.score += 50 # Example score increase for reaching the top
            self.win_sound.play()

        observation = self.get_observation()
        return observation, reward, self.done, {}

    def update_score(self):
        new_lane = self.get_lane_index(self.agent.y)
        if new_lane < self.current_lane:
            self.score += (self.current_lane - new_lane)
            self.current_lane = new_lane

    def get_observation(self):
        observation = {
            "agent": (self.agent.x, self.agent.y),
            "vehicles": [(vehicle.x, vehicle.y) for vehicle in self.vehicles],
            "score": self.score
        }
        return observation

    def reset(self):
        self.agent = Frog() # Re-instantiate Frog class
        self.agent.x = GAME_CONFIG['SCREEN_WIDTH'] // 2 - self.agent.width // 2
        self.agent.y = self.road_bottom - self.agent.height - 10
        self.spawn_vehicles()
        self.done = False
        self.score = 0
        self.current_lane = self.get_lane_index(self.agent.y)
        # Vehicle pool logic removed, no need to return vehicles
        return self.get_observation()

    def close(self):
        pygame.quit()