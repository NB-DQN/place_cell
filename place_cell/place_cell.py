from abc import ABCMeta, abstractmethod

class PlaceCell(object):
    __metaclass__ = ABCMeta

    def __init__(self, environment=None):
        self.__environment = environment

    @abstractmethod
    def move(self, action):
        pass

    @abstractmethod
    def novelty(self):
        pass
