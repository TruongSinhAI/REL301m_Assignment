from src.constants.config import *
from src.utils.resource_manager import ResourceManager
import pygame
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class BaseVehicle:
    def __init__(self, x, y, speed, direction, width=60, height=40, lane_index=0):
        self.x = x
        self.y = y
        self.speed = speed
        self.direction = direction
        self.width = width
        self.height = height
        self.lane_index = lane_index
        
    def update(self, vehicles):
        new_x = self.x + self.speed * self.direction
        can_move = True
        for other in vehicles:
            if other != self and other.lane_index == self.lane_index:
                if self._check_collision(other, new_x):
                    can_move = True
                    break
        if can_move:
            self.x = new_x
        if self.direction == 1 and self.x > GAME_CONFIG['SCREEN_WIDTH'] + 1.5 * self.width:
            self.x = -1.5 * self.width
        elif self.direction == -1 and self.x < -1.5 * self.width:
            self.x = GAME_CONFIG['SCREEN_WIDTH'] + 1.5 * self.width

    def _check_collision(self, other, new_x):
        return (new_x < other.x + other.width and 
                new_x + self.width > other.x and 
                self.y < other.y + other.height and 
                self.y + self.height > other.y)

    def get_bounds(self):
        return (self.x, self.y, self.width, self.height)


class Vehicle(BaseVehicle):
    def __init__(self, x, y, speed, direction, width=60, height=40, lane_index=0):
        super().__init__(x, y, speed, direction, width, height, lane_index)
        self.resource_manager = ResourceManager()
        self.image = self.resource_manager.get_image("vehicle", self.width, self.height)

    def draw(self, surface, camera_offset):
        draw_x = self.x
        draw_y = self.y - camera_offset
        surface.blit(self.image, (draw_x, draw_y))

    def get_rect(self):
        return pygame.Rect(*self.get_bounds())
