import pygame
import os

from src.constants.config import GAME_CONFIG

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
        print(os.getcwd())
        self._load_image("agent", "./assets/images/agent.png")
        self._load_image("vehicle", "./assets/images/vehicle.png")

        # Load sounds
        self._load_sound("crash", "./assets/sounds/crash.ogg")
        self._load_sound("win", "./assets/sounds/win.ogg")

    def _load_image(self, name, path):
        try:
            image = pygame.image.load(path).convert_alpha()
            self.images[name] = image
        except pygame.error as e:
            print(f"Could not load image: {path}", e)
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
