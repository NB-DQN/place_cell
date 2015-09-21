from place_cell import PlaceCell

class DeterministicPlaceCell(PlaceCell):
    def __init__(self, size):
        self.environment_size = size
        self.virtual_coordinate = (0, 0)
        self.history = {}
        self.novelty = 0

    def neighbor(self, action):
        neighbors = [ \
            (self.virtual_coordinate[0] + 1, self.virtual_coordinate[1]    ), \
            (self.virtual_coordinate[0] - 1, self.virtual_coordinate[1]    ), \
            (self.virtual_coordinate[0]    , self.virtual_coordinate[1] + 1), \
            (self.virtual_coordinate[0]    , self.virtual_coordinate[1] - 1)]
        return neighbors[action]

    def validate_action(self, action):
        coordinate = self.neighbor(action)
        return 0 <= coordinate[0] < self.environment_size[0] and \
               0 <= coordinate[1] < self.environment_size[1]

    def move(self, action, step):
        self.virtual_coordinate = self.neighbor(action)
        self._DeterministicPlaceCell__check_novelty()
        return self._DeterministicPlaceCell__check_steps(step)

    def coordinate_id(self):
        return self.virtual_coordinate[0] + self.virtual_coordinate[1] * self.environment_size[0]

    def set_coordinate_id(self, coordinate_id):
        new_x = coordinate_id % self.environment_size[0]
        new_y = (coordinate_id - new_x) / self.environment_size[0]
        self.virtual_coordinate = (new_x, new_y)

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

    def __check_steps(self, step):
        if self.history.setdefault(self.coordinate_id(), step) <= step:
            self.history[self.coordinate_id()] = step
            return True
        else:
            return False
