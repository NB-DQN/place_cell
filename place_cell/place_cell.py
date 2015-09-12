from abc import ABCMeta, abstractmethod

class PlaceCell(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def validate_action(self, action):
        pass

    @abstractmethod
    def virtual_move(self, action):
        pass

    @abstractmethod
    def move(self, action):
        pass

    @abstractmethod
    def coordinate_id(self):
        pass

    @abstractmethod
    def novelty(self):
        pass
