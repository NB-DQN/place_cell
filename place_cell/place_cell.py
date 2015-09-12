from abc import ABCMeta, abstractmethod

class PlaceCell(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def move(self, action):
        pass

    @abstractmethod
    def coordinate_id(self):
        pass

    @abstractmethod
    def novelty(self):
        pass
