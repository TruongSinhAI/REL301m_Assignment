from src.constants.config import *
from src.utils.resource_manager import ResourceManager
import pygame

class BaseFrog:
    def __init__(self):
        self.width = 40
        self.height = 40
        self.x = GAME_CONFIG['SCREEN_WIDTH'] // 2 - self.width // 2
        self.y = 0
        self.speed = 5

    def move(self, dx, dy):
        new_x = self.x + dx * self.speed
        new_y = self.y + dy * self.speed
        self.x = max(0, min(new_x, GAME_CONFIG['SCREEN_WIDTH'] - self.width))
        self.y = new_y

    def get_bounds(self):
        return (self.x, self.y, self.width, self.height)

class Frog(BaseFrog):
    def __init__(self):
        super().__init__()
        self.resource_manager = ResourceManager()
        self.image = self.resource_manager.get_image("agent", self.width, self.height)

    def draw(self, surface, camera_offset):
        draw_x = self.x
        draw_y = self.y - camera_offset
        surface.blit(self.image, (draw_x, draw_y))

    def get_rect(self):
        return pygame.Rect(*self.get_bounds())