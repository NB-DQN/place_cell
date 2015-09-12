from place_cell import PlaceCell

class DeterministicPlaceCell(PlaceCell):
    def move(self, action):
        self._DeterministicPlaceCell__environment.move(action)

    def novelty(self):
        return self._DeterministicPlaceCell__environment.novelty
