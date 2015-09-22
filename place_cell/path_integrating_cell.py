from place_cell import PlaceCell
import os
import chainer
import chainer.functions as F
import pickle
import numpy as np

class PathIntegratingCell(PlaceCell):
    def __init__(self, size):
        super(PathIntegrateCell, self).__init__(size)

        self.offset = 2
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'pi' + str(self.offset) + '.pkl')
        f = open(filename, 'rb')
        self.pretrained_model =  pickle.load(f)
        f.close()

        self.state = self.make_initial_state(batchsize=1, train=False)

    def make_initial_state(self, batchsize=1, train=True):
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

    def move(self, action, precise_coordinate=None):
        if   action == 0:
            action = [1, 0, 0, 0]
        elif action == 1:
            action = [0, 1, 0, 0]
        elif action == 2:
            action = [0, 0, 1, 0]
        elif action == 3:
            action = [0, 0, 0, 1]

        coordinate_one_hot = [0] * 81
        if isinstance(precise_coordinate, tuple):
            current_cid = precise_coordinate[0] + self.environment_size[0] * precise_coordinate[1]
            if current_cid % self.offset == 0 and (self.offset == 2 or (self.offset == 4 and current_cid // self.environment_size[0] % self.offset == 0)):
                coordinate_one_hot[current_cid] = 1
        else:
            current_cid = self.virtual_coordinate[0] + self.virtual_coordinate[1] * self.environment_size[0]
            if current_cid % self.offset == 0 and (self.offset == 2 or (self.offset == 4 and current_cid // self.environment_size[0] % self.offset == 0)):
                coordinate_one_hot[current_cid] = 1
        data = np.array([action + coordinate_one_hot], dtype='float32')
        x = chainer.Variable(data, volatile=True)
        h_in = self.pretrained_model.x_to_h(x) + self.pretrained_model.h_to_h(self.state['h'])
        c, h = F.lstm(self.state['c'], h_in)
        self.state = {'c': c, 'h': h}

        y = self.pretrained_model.h_to_y(h)
        exp_y = np.exp(y.data[0], out=y.data[0])
        softmax_y = exp_y / exp_y.sum(axis=0, keepdims=True)
        cid = softmax_y.argmax()
        self.set_coordinate_id(cid)

        output = np.zeros(self.environment_size[0] * self.environment_size[1], dtype=np.bool)
        output[cid] = 1
        self._PlaceCell__check_novelty(output)

        return True

    def coordinate_id(self):
        return self.virtual_coordinate[0] + self.virtual_coordinate[1] * self.environment_size[0]

    def set_coordinate_id(self, coordinate_id):
        new_x = coordinate_id % self.environment_size[0]
        new_y = (coordinate_id - new_x) / self.environment_size[0]
        self.virtual_coordinate = (new_x, new_y)
