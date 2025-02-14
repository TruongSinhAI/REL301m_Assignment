import pygame
import random
from enum import Enum

# Game Constants (giữ nguyên)
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800
FPS = 60

# Improved Colors (giữ nguyên)
COLORS = {
    'BLACK': (0, 0, 0),
    'WHITE': (255, 255, 255),
    'RED': (220, 50, 50),
    'BLUE': (50, 150, 220),
    'GRASS': (50, 150, 50),
    'GOLD': (255, 215, 0),
    'VEHICLE_LANE': (80, 80, 80),
    'REST_LANE': (150, 150, 150),
    'LIGHT_GRAY': (200, 200, 200),
    'BUTTON_GRAY': (100, 100, 100),
    'BUTTON_HOVER_GRAY': (120, 120, 120)
}

# Game Configuration (giữ nguyên)
GAME_CONFIG = {
    'TOTAL_LANES': 15,
    'LANE_HEIGHT': 60,
    'ROAD_TOP': 80,
    'DASH_LENGTH': 15,
    'VEHICLE_SPAWN_DELAY': 3000,
    'MIN_VEHICLE_SPACING': 200,
    'MIN_VEHICLE_WIDTH': 40,
    'MAX_VEHICLE_WIDTH': 60,
    'MIN_VEHICLE_HEIGHT': 30,
    'MAX_VEHICLE_HEIGHT': 40,
    'MIN_VEHICLE_SPEED': 2,
    'MAX_VEHICLE_SPEED': 4
}

class GameState(Enum): # (giữ nguyên)
    MENU = 1
    PLAYING = 2
    PAUSED = 3
    GAME_OVER = 4

import pygame
import os

class ResourceManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResourceManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not ResourceManager._initialized:
            self.images = {}
            self.sounds = {}
            self._load_resources()
            ResourceManager._initialized = True
    
    def _load_resources(self):
        # Load images
        self._load_image("agent", "./assets/images/agent.png")
        self._load_image("vehicle", "./assets/images/vehicle.png")
        
        # Load sounds
        self._load_sound("crash", "./assets/sounds/crash.ogg")
        self._load_sound("win", "./assets/sounds/win.ogg")
    
    def _load_image(self, name, path):
        try:
            image = pygame.image.load(path).convert_alpha()
            self.images[name] = image
        except pygame.error:
            print(f"Could not load image: {path}")
            # Create a colored rectangle as fallback
            surface = pygame.Surface((40, 40), pygame.SRCALPHA)
            surface.fill((255, 0, 0) if name == "vehicle" else (0, 255, 0))
            self.images[name] = surface
    
    def _load_sound(self, name, path):
        try:
            self.sounds[name] = pygame.mixer.Sound(path)
        except pygame.error:
            print(f"Could not load sound: {path}")
            # Create a silent sound as fallback
            self.sounds[name] = pygame.mixer.Sound(buffer=b'\x00' * 44100)
    
    def get_image(self, name, width=None, height=None):
        image = self.images.get(name)
        if image and (width is not None and height is not None):
            return pygame.transform.scale(image, (width, height))
        return image
    
    def get_sound(self, name):
        return self.sounds.get(name)

class Agent:
    def __init__(self):
        self.width = 40
        self.height = 40
        self.x = SCREEN_WIDTH // 2 - self.width // 2
        self.y = 0
        self.speed = 5
        # Load hình ảnh agent
        # self.image = pygame.image.load("agent.png").convert_alpha()
        # self.image = pygame.transform.scale(self.image, (self.width, self.height)) # Scale ảnh cho vừa kích thước agent
        self.resource_manager = ResourceManager()
        self.image = self.resource_manager.get_image("agent", self.width, self.height)
        # **Ví dụ hoạt ảnh đơn giản (chỉ là minh họa, có thể phức tạp hơn)**
        # self.animation_frames = [
        #     pygame.image.load("agent_frame1.png").convert_alpha(),
        #     pygame.image.load("agent_frame2.png").convert_alpha()
        # ]
        # self.animation_frames = [pygame.transform.scale(frame, (self.width, self.height)) for frame in self.animation_frames] # Scale các frame
        # self.current_frame = 0
        # self.animation_timer = 0
        # self.animation_speed = 100 # milliseconds per frame

    def move(self, dx, dy): # (giữ nguyên logic di chuyển)
        new_x = self.x + dx * self.speed
        new_y = self.y + dy * self.speed
        self.x = max(0, min(new_x, SCREEN_WIDTH - self.width))
        self.y = new_y

    def draw(self, surface, camera_offset):
        draw_x = self.x
        draw_y = self.y - camera_offset
        # Thay vì vẽ hình chữ nhật, vẽ hình ảnh agent
        surface.blit(self.image, (draw_x, draw_y))
        # **Nếu có hoạt ảnh, thay vì dòng trên:**
        # surface.blit(self.animation_frames[self.current_frame], (draw_x, draw_y))

    def get_rect(self): # (giữ nguyên)
        return pygame.Rect(self.x, self.y, self.width, self.height)

    # **Ví dụ hàm update hoạt ảnh (nếu bạn thêm hoạt ảnh)**
    # def update_animation(self, delta_time):
    #     self.animation_timer += delta_time
    #     if self.animation_timer > self.animation_speed:
    #         self.animation_timer -= self.animation_speed
    #         self.current_frame = (self.current_frame + 1) % len(self.animation_frames)

class Vehicle:
    def __init__(self, x, y, speed, direction, width=60, height=40, lane_index=0):
        self.x = x
        self.y = y
        self.speed = speed
        self.direction = direction
        self.width = width
        self.height = height
        self.lane_index = lane_index
        # Load hình ảnh xe
        # self.image = pygame.image.load("vehicle.png").convert_alpha() # Mặc định xe đỏ
        # self.image = pygame.transform.scale(self.image, (self.width, self.height)) # Scale ảnh cho vừa kích thước xe
        self.resource_manager = ResourceManager()
        self.image = self.resource_manager.get_image("vehicle", self.width, self.height)
        
    def update(self, vehicles): # (giữ nguyên logic update xe)
        new_x = self.x + self.speed * self.direction
        temp_rect = pygame.Rect(new_x, self.y, self.width, self.height)
        can_move = True
        for other in vehicles:
            if other != self and other.lane_index == self.lane_index:
                if temp_rect.colliderect(pygame.Rect(other.x, other.y, other.width, other.height)):
                    can_move = True
                    break
        if can_move:
            self.x = new_x
        if self.direction == 1 and self.x > SCREEN_WIDTH + 1.5 * self.width:
            self.x = -1.5 * self.width
        elif self.direction == -1 and self.x < -1.5 * self.width:
            self.x = SCREEN_WIDTH + 1.5 * self.width

    def draw(self, surface, camera_offset):
        draw_x = self.x
        draw_y = self.y - camera_offset
        # Thay vì vẽ hình chữ nhật, vẽ hình ảnh xe
        surface.blit(self.image, (draw_x, draw_y))

    def get_rect(self): # (giữ nguyên)
        return pygame.Rect(self.x, self.y, self.width, self.height)

class Button: # (giữ nguyên class Button, nếu bạn đã chỉnh sửa nút bấm hình ảnh ở câu hỏi trước, hãy giữ lại phiên bản đó)
    def __init__(self, text, x, y, width, height, color, hover_color, font, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.text = text
        self.font = font
        self.text_surface = font.render(text, True, COLORS['WHITE'])
        self.text_rect = self.text_surface.get_rect(center=self.rect.center)
        self.action = action
        self.hovered = False

    def draw(self, surface):
        current_color = self.hover_color if self.hovered else self.color
        pygame.draw.rect(surface, current_color, self.rect)
        surface.blit(self.text_surface, self.text_rect)

    def check_hover(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

    def check_click(self, mouse_pos):
        if self.hovered and self.action:
            return self.action()
        return None

class VehiclePool:
    def __init__(self, initial_size=50):
        self.available_vehicles = []
        self.active_vehicles = set()
        self.resource_manager = ResourceManager()
        
        # Pre-create vehicles
        for _ in range(initial_size):
            vehicle = Vehicle(0, 0, 0, 1)
            self.available_vehicles.append(vehicle)
    
    def get_vehicle(self, x, y, speed, direction, width, height, lane_index):
        if self.available_vehicles:
            vehicle = self.available_vehicles.pop()
        else:
            vehicle = Vehicle(0, 0, 0, 1)
        
        # Reset vehicle properties
        vehicle.x = x
        vehicle.y = y
        vehicle.speed = speed
        vehicle.direction = direction
        vehicle.width = width
        vehicle.height = height
        vehicle.lane_index = lane_index
        vehicle.image = self.resource_manager.get_image("vehicle", width, height)
        
        self.active_vehicles.add(vehicle)
        return vehicle
    
    def return_vehicle(self, vehicle):
        if vehicle in self.active_vehicles:
            self.active_vehicles.remove(vehicle)
            self.available_vehicles.append(vehicle)

class CarCrossingEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Car Crossing Game")
        self.clock = pygame.time.Clock()
        self.fps_font = pygame.font.SysFont("Helvetica", 24)
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

        self.agent = Agent()
        self.agent.y = self.road_bottom - self.agent.height - 10

        self.vehicle_pool = VehiclePool()
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
        menu_button_y_start = SCREEN_HEIGHT // 2 - (button_height * 2) // 2
        button_x = SCREEN_WIDTH // 2 - button_width // 2

        self.manual_button = Button("Manual Mode", button_x, menu_button_y_start, button_width, button_height, COLORS['BUTTON_GRAY'], COLORS['BUTTON_HOVER_GRAY'], self.menu_font, action=lambda: self.set_mode("manual"))
        self.ai_button = Button("AI Mode", button_x, menu_button_y_start + button_height + 20, button_width, button_height, COLORS['BUTTON_GRAY'], COLORS['BUTTON_HOVER_GRAY'], self.menu_font, action=lambda: self.set_mode("ai"))
        self.menu_buttons = [self.manual_button, self.ai_button]

        # Game Over Buttons (giữ nguyên hoặc dùng nút bấm hình ảnh nếu đã chỉnh sửa)
        game_over_button_y_start = SCREEN_HEIGHT // 2 + 50
        self.restart_button = Button("Restart", button_x, game_over_button_y_start, button_width, button_height, COLORS['BUTTON_GRAY'], COLORS['BUTTON_HOVER_GRAY'], self.instruction_font, action=lambda: self.restart_game())
        self.menu_return_button = Button("Menu", button_x, game_over_button_y_start + button_height + 10, button_width, button_height, COLORS['BUTTON_GRAY'], COLORS['BUTTON_HOVER_GRAY'], self.instruction_font, action=lambda: self.return_to_menu())
        self.game_over_buttons = [self.restart_button, self.menu_return_button]

        self.mode = "menu"

        # **Âm thanh**
        pygame.mixer.init() # Khởi tạo mixer (nếu chưa chắc chắn đã khởi tạo trước đó)
        # self.collision_sound = pygame.mixer.Sound("crash.mp3") # Load âm thanh va chạm
        # self.win_sound = pygame.mixer.Sound("win.mp3") # Load âm thanh chiến thắng
        # Initialize ResourceManager
        self.resource_manager = ResourceManager()
        # Load sounds once
        self.collision_sound = self.resource_manager.get_sound("crash")
        self.win_sound = self.resource_manager.get_sound("win")

        # **(Tùy chọn) Amy thanh động cơ xe hơi**
        # self.car_engine_sound = pygame.mixer.Sound("car_engine.wav")
        # self.car_engine_channel = pygame.mixer.Channel(0) # Chọn channel để phát âm thanh động cơ
        # self.car_engine_channel.set_volume(0.5) # Giảm âm lượng động cơ
        # self.play_car_engine_sound() # Bắt đầu phát âm thanh động cơ khi game chạy

    '''# **(Tùy chọn) Hàm phát âm thanh động cơ (loop)**
    # def play_car_engine_sound(self):
    #     if self.car_engine_sound and not self.car_engine_channel.get_busy():
    #         self.car_engine_channel.play(self.car_engine_sound, loops=-1) # loops=-1 để lặp vô hạn

    # **(Tùy chọn) Hàm dừng âm thanh động cơ**
    # def stop_car_engine_sound(self):
    #     if self.car_engine_channel.get_busy():
    #         self.car_engine_channel.stop()
    '''

    def set_mode(self, mode_str): # (giữ nguyên)
        self.mode = mode_str
        self.game_state = GameState.PLAYING
        # **Bắt đầu phát âm thanh động cơ khi vào chế độ chơi**
        # self.play_car_engine_sound()

    def restart_game(self): # (giữ nguyên)
        self.reset()
        self.game_state = GameState.PLAYING
        # **Bắt đầu phát âm thanh động cơ khi restart game**
        # self.play_car_engine_sound()

    def return_to_menu(self): # (giữ nguyên)
        self.game_state = GameState.MENU
        self.mode = "menu"
        # **Dừng âm thanh động cơ khi về menu**
        # self.stop_car_engine_sound()

    def spawn_vehicles(self):
        self.vehicles = []
        lane_width = SCREEN_WIDTH + 300
        for lane_index in range(self.total_lanes):
            if lane_index % 3 == 2:
                continue
            lane_y = self.road_top + 20 + lane_index * self.lane_height
            direction = 1 if lane_index % 2 == 0 else -1
            available_space = lane_width
            current_x = 0
            while available_space > GAME_CONFIG['MIN_VEHICLE_SPACING']:
                width = random.randint(self.min_vehicle_width, self.max_vehicle_width)
                height = random.randint(self.min_vehicle_height, self.max_vehicle_height)
                speed = random.randint(self.min_vehicle_speed, self.max_vehicle_speed) + random.random()
                x_position = current_x if direction == 1 else SCREEN_WIDTH + current_x
                
                # Use vehicle pool
                vehicle = self.vehicle_pool.get_vehicle(
                    x_position, lane_y, speed, direction, 
                    width, height, lane_index
                )
                self.vehicles.append(vehicle)
                vehicle_spacing = width + GAME_CONFIG['MIN_VEHICLE_SPACING']
                available_space -= vehicle_spacing
                current_x += vehicle_spacing
        self.vehicles = []
        lane_width = SCREEN_WIDTH + 300
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
                    x_position = SCREEN_WIDTH + current_x
                vehicle = Vehicle(x_position, lane_y, speed, direction, width, height, lane_index)
                vehicles_in_lane.append(vehicle)
                vehicle_spacing = width + GAME_CONFIG['MIN_VEHICLE_SPACING']
                available_space -= vehicle_spacing
                current_x += vehicle_spacing
            self.vehicles.extend(vehicles_in_lane)

    def render(self, mode="manual"): # (chỉnh sửa để update animation nếu có)
        if self.game_state == GameState.MENU:
            self.render_menu()
            return
        elif self.game_state == GameState.GAME_OVER:
            self.render_game_over()
            return
        elif self.game_state == GameState.PAUSED:
            self.render_pause_screen()
            return

        camera_offset = max(0, min(self.agent.y - (SCREEN_HEIGHT - 150), self.road_bottom - SCREEN_HEIGHT))

        self.screen.fill(COLORS['GRASS'])

        self._draw_lanes(camera_offset)

        finish_line_y = self.road_top - camera_offset
        pygame.draw.line(self.screen, COLORS['GOLD'], (0, finish_line_y), (SCREEN_WIDTH, finish_line_y), 5)

        # **Update animation frame cho agent (nếu có hoạt ảnh)**
        # delta_time = self.clock.get_time() # Lấy thời gian từ frame trước
        # self.agent.update_animation(delta_time)

        self.agent.draw(self.screen, camera_offset)
        for vehicle in self.vehicles:
            vehicle.draw(self.screen, camera_offset)

        self._draw_game_info(mode)

        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_game_info(self, mode): # (giữ nguyên)
        fps = self.clock.get_fps()
        fps_text = self.fps_font.render(f"FPS: {int(fps)}", True, COLORS['WHITE'])
        self.screen.blit(fps_text, (10, 40))

        mode_text = self.score_font.render(f"Mode: {mode.upper()}", True, COLORS['GOLD'])
        score_text = self.score_font.render(f"Score: {self.score}", True, COLORS['GOLD'])
        self.screen.blit(mode_text, (10, 10))
        self.screen.blit(score_text, (SCREEN_WIDTH - score_text.get_width() - 10, 10))

    def render_menu(self): # (giữ nguyên)
        self.screen.fill(COLORS['GRASS'])
        title_text = self.title_font.render("Car Crossing Game", True, COLORS['WHITE'])
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3))
        self.screen.blit(title_text, title_rect)
        for button in self.menu_buttons:
            button.draw(self.screen)
        pygame.display.flip()

    def render_game_over(self): # (chỉnh sửa để phát âm thanh game over và thấy hình phía sau)
        # **Render game scene phía sau trước**
        camera_offset = max(0, min(self.agent.y - (SCREEN_HEIGHT - 150), self.road_bottom - SCREEN_HEIGHT))
        self.screen.fill(COLORS['GRASS'])
        self._draw_lanes(camera_offset)
        finish_line_y = self.road_top - camera_offset
        pygame.draw.line(self.screen, COLORS['GOLD'], (0, finish_line_y), (SCREEN_WIDTH, finish_line_y), 5)
        self.agent.draw(self.screen, camera_offset)
        for vehicle in self.vehicles:
            vehicle.draw(self.screen, camera_offset)
        self._draw_game_info(self.mode) # Cập nhật thông tin game (FPS, Score, Mode) trên nền game over

        # **Vẽ lớp overlay mờ lên trên**
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(128)
        self.screen.blit(overlay, (0, 0))

        # **Vẽ chữ "Game Over" và điểm số**
        game_over_text = self.game_over_font.render("Game Over!", True, COLORS['WHITE'])
        score_text = self.score_font.render(f"Score: {self.score}", True, COLORS['GOLD'])
        game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3))
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30))
        self.screen.blit(game_over_text, game_over_rect)
        self.screen.blit(score_text, score_rect)

        # **Vẽ các nút bấm Game Over**
        for button in self.game_over_buttons:
            button.draw(self.screen)

        pygame.display.flip()

    def render_pause_screen(self): # (giữ nguyên)
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(128)
        self.screen.blit(overlay, (0, 0))
        pause_text = self.game_over_font.render("PAUSED", True, COLORS['WHITE'])
        continue_text = self.instruction_font.render("Press P to Continue", True, COLORS['WHITE'])
        pause_rect = pause_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3))
        continue_rect = continue_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(pause_text, pause_rect)
        self.screen.blit(continue_text, continue_rect)
        pygame.display.flip()

    def _draw_lanes(self, camera_offset): # (giữ nguyên)
        vehicle_lane_color = COLORS['VEHICLE_LANE']
        rest_lane_color = COLORS['REST_LANE']
        dash_color = COLORS['LIGHT_GRAY']
        dash_length = GAME_CONFIG['DASH_LENGTH']
        lane_height = GAME_CONFIG['LANE_HEIGHT']

        for lane_index in range(self.total_lanes):
            lane_y = self.road_top + 20 + lane_index * lane_height - camera_offset
            lane_rect = pygame.Rect(0, lane_y, SCREEN_WIDTH, lane_height)
            if lane_index % 3 == 2:
                pygame.draw.rect(self.screen, rest_lane_color, lane_rect)
            else:
                pygame.draw.rect(self.screen, vehicle_lane_color, lane_rect)
                for x in range(0, SCREEN_WIDTH, 2 * dash_length):
                    pygame.draw.line(self.screen, dash_color, (x, lane_y + lane_height // 2), (x + dash_length, lane_y + lane_height // 2), 3)


    def get_lane_index(self, y_pos): # (giữ nguyên)
        lane = int((y_pos - (self.road_top + 20)) // self.lane_height)
        return max(lane, 0)

    def step(self, action): # (chỉnh sửa để thêm âm thanh va chạm và chiến thắng)
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
            vehicle.update(self.vehicles)

        self.update_score()

        reward = 0
        agent_rect = self.agent.get_rect()

        for vehicle in self.vehicles:
            if agent_rect.colliderect(vehicle.get_rect()):
                self.done = True
                reward = -10
                # **Phát âm thanh va chạm khi va chạm**
                self.collision_sound.play()
                # **Dừng âm thanh động cơ khi va chạm (nếu có)**
                # self.stop_car_engine_sound()
                break

        if self.agent.y <= self.road_top:
            self.done = True
            reward = 10
            # **Phát âm thanh chiến thắng khi về đích**
            self.win_sound.play()
            # **Dừng âm thanh động cơ khi thắng (nếu có)**
            # self.stop_car_engine_sound()

        observation = self.get_observation()
        return observation, reward, self.done, {}

    def update_score(self): # (giữ nguyên)
        new_lane = self.get_lane_index(self.agent.y)
        if new_lane < self.current_lane:
            self.score += (self.current_lane - new_lane)
            self.current_lane = new_lane

    def get_observation(self): # (giữ nguyên)
        observation = {
            "agent": (self.agent.x, self.agent.y),
            "vehicles": [(vehicle.x, vehicle.y) for vehicle in self.vehicles],
            "score": self.score
        }
        return observation

    def reset(self): # (giữ nguyên)
        self.agent = Agent()
        self.agent.x = SCREEN_WIDTH // 2 - self.agent.width // 2
        self.agent.y = self.road_bottom - self.agent.height - 10
        self.spawn_vehicles()
        self.done = False
        self.score = 0
        self.current_lane = self.get_lane_index(self.agent.y)
        for vehicle in self.vehicles:
            self.vehicle_pool.return_vehicle(vehicle)
        return self.get_observation()

    def close(self): # (giữ nguyên)
        pygame.quit()

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

def main(): # (chỉnh sửa để dừng âm thanh động cơ khi thoát game)
    env = CarCrossingEnv()
    running = True
    mouse_pos = (0, 0)

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
                action = simple_ai_policy(env.agent, env.vehicles)

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