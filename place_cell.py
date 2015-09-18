import train
import pickle
import numpy as np
import chainer
import chainer.functions as F

class PlaceCell:
    def __init__(self, maze_size_x=9, maze_size_y=9, pretrained = True):
        self.maze_size_x = maze_size_x
        self.maze_size_y = maze_size_y

        if pretrained == True:
            f = open('pretrained_model_'+str(maze_size_x)+'_'+str(maze_size_y)+'.pkl', 'rb')
            self.pretrained_model =  pickle.load(f)
            f.close()
        else:    
            self.pretrained_model = train.pretrain(maze_size_x, maze_size_y)

        self.state = train.make_initial_state(batchsize=1, train=False)
        self.location_1d = 0
        self.history = []
        
    def move(self, direction):
    	data = np.array([direction], dtype='float32')
        x = chainer.Variable(data, volatile=True)
        h_in = self.pretrained_model.x_to_h(x) + self.pretrained_model.h_to_h(self.state['h'])  
        c, h = F.lstm(self.state['c'], h_in)
        self.y = self.pretrained_model.h_to_y(h)
        self.state = {'c': c, 'h': h}
        self.location_one_hot = self.y.data.reshape(len(self.y), -1)
        self.location_1d = self.location_one_hot.argmax()
        
        # store history
        self.history.append(self.location_1d)
         
    def current_coordinate(self):
    	self.location_2d = [self.location_1d % self.maze_size_x, 
    	        self.location_1d // self.maze_size_x]
        return self.location_2d
        
    def novelty(self):
        if self.location_1d in self.history:
    	    return 0
        else:
        	return 1