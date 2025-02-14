from src.utils.resource_manager import ResourceManager
from src.agents.vehicle import *


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
