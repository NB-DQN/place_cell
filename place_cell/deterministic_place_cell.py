from place_cell import PlaceCell

class DeterministicPlaceCell(PlaceCell):
    def __init__(self, environment):
        self.__environment = environment
        self.virtual_coordinate = (0, 0)

    def validate_action(self, action):
        return 0 <= virtual_coordinate[0] <= 8 and 0 <= virtual_coordinate[1] <= 8

    def virtual_move(self, action):
        neighbor = [ \
            (self.current_coordinate[0] + 1, self.current_coordinate[1]    ), \
            (self.current_coordinate[0] - 1, self.current_coordinate[1]    ), \
            (self.current_coordinate[0]    , self.current_coordinate[1] + 1), \
            (self.current_coordinate[0]    , self.current_coordinate[1] - 1)]
        self.virtual_coordinate = neighbor[direction]

    def move(self, action):
        self._DeterministicPlaceCell__environment.move(action)
        virtual_coordinate = self._DeterministicPlaceCell__environment.current_coordinate

    def coordinate_id(self):
        return virtual_coordinate[0] + virtual_coordinate[1] * self._DeterministicPlaceCell__environment.size[1]

    def novelty(self):
        return self._DeterministicPlaceCell__environment.novelty
