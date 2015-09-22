from place_cell import PlaceCell
import os
import chainer
import chainer.functions as F
import pickle
import numpy as np

class VisualPlaceCell(PlaceCell):
    def __init__(self, size):
        super(VisualPlaceCell, self).__init__(size)

        self.n_units = 60

        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'vpc_rnn_' + str(self.n_units) + '.pkl')
        f = open(filename, 'rb')
        self.pretrained_model =  pickle.load(f)
        f.close()

        self.state = self.make_initial_state(batchsize=1, train=False)

        filename = os.path.join(dirname, 'vpc_svm_' + str(self.n_units) + '.pkl')
        f = open(filename, 'rb')
        self.clf =  pickle.load(f)
        f.close()

        self.predicted_visual_image = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

    def make_initial_state(self, batchsize=1, train=True):
        return { name: chainer.Variable(np.zeros((batchsize, self.n_units), dtype=np.float32), volatile=not train) for name in ('c', 'h') }

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

    def move(self, action, visual_image=None):
        if   action == 0:
            action = [1, 0, 0, 0]
        elif action == 1:
            action = [0, 1, 0, 0]
        elif action == 2:
            action = [0, 0, 1, 0]
        elif action == 3:
            action = [0, 0, 0, 1]

        if visual_image is None:
            data = np.array([action + self.predicted_visual_image.tolist()], dtype='float32')
        else:
            data = np.array([action + visual_image.tolist()], dtype='float32')
        x = chainer.Variable(data, volatile=True)
        h_in = self.pretrained_model.x_to_h(x) + self.pretrained_model.h_to_h(self.state['h'])
        c, h = F.lstm(self.state['c'], h_in)
        self.state = {'c': c, 'h': h}

        y = self.pretrained_model.h_to_y(h)
        sigmoid_y = 1 / (1 + np.exp(-y.data))
        self.predicted_visual_image = np.round((np.sign(sigmoid_y - 0.5) + 1) / 2)[0]

        cid = self.hidden_to_coordinates(h.data[0])
        self.set_coordinate_id(cid)

        output = np.zeros(self.environment_size[0] * self.environment_size[1], dtype=np.bool)
        output[self.coordinate_id()] = 1
        self._PlaceCell__check_novelty(output)

        return True

    def coordinate_id(self):
        return self.virtual_coordinate[0] + self.virtual_coordinate[1] * self.environment_size[0]

    def set_coordinate_id(self, coordinate_id):
        new_x = coordinate_id % self.environment_size[0]
        new_y = (coordinate_id - new_x) / self.environment_size[0]
        self.virtual_coordinate = (new_x, new_y)

    def hidden_to_coordinates(self, hidden):
        return self.clf.predict(hidden)[0]

