from abc import ABCMeta, abstractmethod
import numpy as np

class PlaceCell(object):
    __metaclass__ = ABCMeta

    def __init__(self, size):
        self.environment_size = size
        self.virtual_coordinate = (0, 0)
        self.novelty = 0
        self.novelty_filter = np.zeros(size[0] * size[1], dtype=np.bool)

    def __check_novelty(self, output):
        self.novelty = 10 if (output & np.logical_not(self.novelty_filter)).any() else 0
        self.novelty_filter &= output

    @abstractmethod
    def validate_action(self, action):
        pass

    @abstractmethod
    def move(self, action):
        pass

    @abstractmethod
    def coordinate_id(self):
        pass
