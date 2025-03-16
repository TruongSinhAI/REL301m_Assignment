import pygame
from src.constants.config import COLORS
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Button:
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
