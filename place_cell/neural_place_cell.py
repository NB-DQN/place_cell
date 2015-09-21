from place_cell import PlaceCell
import os
import chainer

def make_initial_state(batchsize=1, train=True):
    return {name: chainer.Variable(np.zeros((batchsize, 25),
                                             dtype=np.float32),
                                   volatile=not train)
            for name in ('c', 'h')}
            
class NeuralPlaceCell(PlaceCell):
    def __init__(self, maze_size = (9,9)):
        self.maze_size_x = maze_size[0]
        self.maze_size_y = maze_size[1]

        f = open('pretrained_model_'+str(maze_size_x)+'_'+str(maze_size_y)+'.pkl', 'rb')
        self.pretrained_model =  pickle.load(f)
        f.close()
        
        os.chdir('/Users/UkitaJumpei/Desktop/NB-DQN/place_cell/trainer/path_integrator')
        import train_practice

        self.state = make_initial_state(batchsize=1, train=False)
        self.location_1d = 0
        self.history = []

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
    	data = np.array([action], dtype='float32')
        x = chainer.Variable(data, volatile=True)
        h_in = self.pretrained_model.x_to_h(x) + self.pretrained_model.h_to_h(self.state['h'])  
        c, h = F.lstm(self.state['c'], h_in)
        y = self.pretrained_model.h_to_y(h)
        self.state = {'c': c, 'h': h}
        exp_y = np.exp(y.data[0], out=y.data[0])
        softmax_y /= exp_y.sum(axis=1, keepdims=True)            
        location_1d = softmax_y.argmax()
        
        # store history
        self.history.append(location_1d)

    def coordinate_id(self):
        pass

    def novelty(self):
        pass
