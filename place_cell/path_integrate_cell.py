from place_cell import PlaceCell
import os
import chainer

class PathIntegrateCell(PlaceCell):
    def __init__(self, size):
        self.environment_size = size
        self.virtual_coordinate = (0, 0)
        self.history = []

        f = open('pretrained_model_'+str(size[0])+'_'+str(size[1])+'.pkl', 'rb')
        self.pretrained_model =  pickle.load(f)
        f.close()

        self.state = self.make_initial_state(batchsize=1, train=False)

    def make_initial_state(batchsize=1, train=True):
        return { name: chainer.Variable(np.zeros((batchsize, 25), dtype=np.float32), volatile=not train) for name in ('c', 'h') }

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

    def move(self, action):
        if   action == 0:
            action = [1, 0, 0, 0]
        elif action == 1:
            action = [0, 1, 0, 0]
        elif action == 2:
            action = [0, 0, 1, 0]
        elif action == 3:
            action = [0, 0, 0, 1]

        data = np.array([[0] * 81 + action], dtype='float32')
        x = chainer.Variable(data, volatile=True)
        h_in = self.pretrained_model.x_to_h(x) + self.pretrained_model.h_to_h(self.state['h'])
        c, h = F.lstm(self.state['c'], h_in)
        y = self.pretrained_model.h_to_y(h)
        self.state = {'c': c, 'h': h}
        exp_y = np.exp(y.data[0], out=y.data[0])
        softmax_y = exp_y / exp_y.sum(axis=1, keepdims=True)
        cid = softmax_y.argmax()

        self.history.append(cid)

        self._DeterministicPlaceCell__check_novelty()

        self.set_coordinate_id(cid)

    def __check_novelty(self):
        flag = False
        for cid in self.history:
            if cid == self.coordinate_id():
                flag = True
                break
        if flag:
            self.novelty = 0
        else:
            self.novelty = 10
            self.history.append(self.coordinate_id())

    def coordinate_id(self):
        return self.virtual_coordinate[0] + self.virtual_coordinate[1] * self.environment_size[0]

    def set_coordinate_id(self, coordinate_id):
        new_x = coordinate_id % self.environment_size[0]
        new_y = (coordinate_id - new_x) / self.environment_size[0]
        self.virtual_coordinate = (new_x, new_y)
