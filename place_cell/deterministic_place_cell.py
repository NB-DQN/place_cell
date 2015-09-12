from place_cell import PlaceCell

class DeterministicPlaceCell(PlaceCell):
    def __init__(self, environment):
        self.__environment = environment

    def move(self, action):
        self._DeterministicPlaceCell__environment.move(action)

    def coordinate_id(self):
        coordinate = self._DeterministicPlaceCell__environment.current_coordinate
        return coordinate[0] + coordinate[1] * self._DeterministicPlaceCell__environment.size[1]

    def novelty(self):
        return self._DeterministicPlaceCell__environment.novelty
