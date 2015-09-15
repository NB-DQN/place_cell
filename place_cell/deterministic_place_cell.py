from place_cell import PlaceCell

class DeterministicPlaceCell(PlaceCell):
    def __init__(self):
        self.virtual_coordinate = (0, 0)
        self.history = {}
        self.novelty = 0

    def validate_action(self, action):
        return 0 <= self.virtual_coordinate[0] <= 8 and 0 <= self.virtual_coordinate[1] <= 8

    def move(self, action):
        neighbor = [ \
            (self.virtual_coordinate[0] + 1, self.virtual_coordinate[1]    ), \
            (self.virtual_coordinate[0] - 1, self.virtual_coordinate[1]    ), \
            (self.virtual_coordinate[0]    , self.virtual_coordinate[1] + 1), \
            (self.virtual_coordinate[0]    , self.virtual_coordinate[1] - 1)]
        self.virtual_coordinate = neighbor[action]
        self._DeterministicPlaceCell__check_novelty()

    def coordinate_id(self):
        return self.virtual_coordinate[0] + self.virtual_coordinate[1] * self._DeterministicPlaceCell__environment.size[1]

    def __check_novelty(self):
        flag = False
        for coordinate in self.history.keys():
            if coordinate == self.coordinate_id():
                flag = True
                break
        if flag:
            self.novelty = 0
        else:
            self.novelty = 10
            self.history.append(self.coordinate_id())
