from place_cell import PlaceCell

class DeterministicPlaceCell(PlaceCell):
    def __init__(self, size):
        self.environment_size = size
        self.virtual_coordinate = (0, 0)
        self.history = {}
        self.novelty = 0

    def validate_action(self, action):
        return 0 <= self.virtual_coordinate[0] <= 8 and 0 <= self.virtual_coordinate[1] <= 8

    def move(self, action, step):
        neighbor = [ \
            (self.virtual_coordinate[0] + 1, self.virtual_coordinate[1]    ), \
            (self.virtual_coordinate[0] - 1, self.virtual_coordinate[1]    ), \
            (self.virtual_coordinate[0]    , self.virtual_coordinate[1] + 1), \
            (self.virtual_coordinate[0]    , self.virtual_coordinate[1] - 1)]
        self.virtual_coordinate = neighbor[action]
        self._DeterministicPlaceCell__check_novelty(step)

    def coordinate_id(self):
        return self.virtual_coordinate[0] + self.virtual_coordinate[1] * self.environment_size[1]

    def set_coordinate_id(self, coordinate_id):
        self.virtual_coordinate[0] = coordinate_id % self.environment_size[1]
        self.virtual_coordinate[1] = (coordinate_id - self.virtual_coordinate[0]) / self.environment_size[1]

    def __check_novelty(self):
        flag = False
        for coordinate in self.history.keys():
            if coordinate == self.coordinate_id():
                flag = True
                break
        if flag:
            self.novelty = 0
            if self.history[self.coordinate_id()] < step:
                self.history[self.coordinate_id()] = step
        else:
            self.novelty = 10
            self.history[self.coordinate_id()] = step
