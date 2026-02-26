from typing import Tuple
import random

class Param:
    def __init__(self, threshold: Tuple[float, float]):
        self.threshold = threshold
        self.value = random.uniform(*threshold)

    def get_param(self):
        return self.value
    
    def set_param(self, value):
        low, high = self.threshold
        self.value = max(min(value, high), low)

    def get_threshold(self):
        return self.threshold