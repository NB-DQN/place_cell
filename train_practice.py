"""
Pre-training the place cell

Model
LSTM with one hidden layer
I don't know if truncated BPTT or gradient clip are necessary here
"""

import argparse
import math
import sys
import time
import random
import pickle

import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

# set parameters
n_epoch = 100000 # 50000 # 1000000 # number of epochs
n_units = 25 # number of units per layer, len(train)=5 -> 20 might be the best
batchsize = 1 # minibatch size
bprop_len = 1 # length of truncated BPTT
valid_len = n_epoch // 100 # 1000 # epoch on which accuracy and perp are calculated
grad_clip = 5 # gradient norm threshold to clip
maze_size_x = 9
maze_size_y = 9

train_data_length = [20, 100]
offset_timing = 1


# GPU
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')                    
args = parser.parse_args()
mod = cuda.cupy if args.gpu >= 0 else np

# generate dataset
"""
input data: direction (2D)
output target: coordinate (converted to 1D)
"""
def generate_seq(seq_length, maze_size_x, maze_size_y):
    directions = []
    locations_1d = [] # 1D coorinate

    current = (0, 0) # 2D coordinate
    for i in range(0, seq_length):
    	direction_choice = [[0, 0.5], [1, 0.5], [0.5, 0], [0.5, 1]] # old code: [[0.5, 0], [1, 0], [0, 0.5], [0, 1]]
        if current[0] == 0:
            direction_choice.remove([0, 0.5])
        if current[0] == maze_size_x-1:
            direction_choice.remove([1, 0.5])
        if current[1] == 0:
            direction_choice.remove([0.5, 0])
        if current[1] == maze_size_y-1:
            direction_choice.remove([0.5, 1])
        direction = random.choice(direction_choice)
        
        if   direction == [0, 0.5]:
            current = (current[0] - 1, current[1]    )
        elif direction == [1, 0.5]:
            current = (current[0] + 1, current[1]    )
        elif direction == [0.5, 0]:
            current = (current[0]    , current[1] - 1)
        elif direction == [0.5, 1]:
            current = (current[0]    , current[1] + 1)
        
        directions.append(direction)
        locations_1d.append(current[0]+current[1]*maze_size_x)
        
    # directions = np.array(directions, dtype='float32')
    # locations_1d = np.array(locations_1d, dtype='int32')
    return directions, locations_1d

def generate_seq_remote(seq_length, maze_size_x, maze_size_y):
    directions = []
    locations_1d = [] # 1D coorinate

    current = (0, 0) # 2D coordinate
    for i in range(0, seq_length):
    	direction_choice = [[0, 0.5], [1, 0.5], [0.5, 0], [0.5, 1]] # old code: [[0.5, 0], [1, 0], [0, 0.5], [0, 1]]
        if current[0] == 0:
            direction_choice.remove([0, 0.5])
        if current[0] == maze_size_x-1:
            direction_choice.remove([1, 0.5])
        if current[1] == 0:
            direction_choice.remove([0.5, 0])
        if current[1] == maze_size_y-1:
            direction_choice.remove([0.5, 1])
        
        if current[0] == 4 and current[1] <= 4:
            threshold = 0.1
            if random.random() > threshold:
                direction_choice.remove([0, 0.5])                    
        if curent[1] == 4 and current[0] <= 4:
            threshold = 0.1
            if random.random() > threshold:
                direction_choice.remove([0.5, 0])
                    
        direction = random.choice(direction_choice)
        
        if   direction == [0, 0.5]:
            current = (current[0] - 1, current[1]    )
        elif direction == [1, 0.5]:
            current = (current[0] + 1, current[1]    )
        elif direction == [0.5, 0]:
            current = (current[0]    , current[1] - 1)
        elif direction == [0.5, 1]:
            current = (current[0]    , current[1] + 1)
        
        directions.append(direction)
        locations_1d.append(current[0]+current[1]*maze_size_x)
        
    # directions = np.array(directions, dtype='float32')
    # locations_1d = np.array(locations_1d, dtype='int32')
    return directions, locations_1d    
    
    
# validation dataset
valid_data, valid_targets = generate_seq(100, maze_size_x, maze_size_y)

# test dataset
test_data, test_targets = generate_seq(100, maze_size_x, maze_size_y)

# model
model = chainer.FunctionSet(
    x_to_h = F.Linear(4, n_units * 4),
    h_to_h = F.Linear(n_units, n_units * 4),
    h_to_y = F.Linear(n_units, maze_size_x * maze_size_y))
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    
# optimizer
optimizer = optimizers.SGD(lr=1.)
optimizer.setup(model.collect_parameters())


# one-step forward propagation
def forward_one_step(data, targets, state, train=True):
    # if args.gpu >= 0:
    #     data = cuda.to_gpu(data)
    #     targets = cuda.to_gpu(targets)
    x = chainer.Variable(data, volatile=not train)
    t = chainer.Variable(targets, volatile=not train)
    h_in = model.x_to_h(x) + model.h_to_h(state['h'])
    c, h = F.lstm(state['c'], h_in)
    y = model.h_to_y(h)
    state = {'c': c, 'h': h}
    return state, F.softmax_cross_entropy(y, t), F.accuracy(y, t)

# initialize hidden state
def make_initial_state(batchsize=batchsize, train=True):
    return {name: chainer.Variable(mod.zeros((batchsize, n_units),
                                             dtype=np.float32),
                                   volatile=not train)
            for name in ('c', 'h')}
             
# evaluation
def evaluate(data, targets, test=False):
    sum_accuracy = mod.zeros(())
    state = make_initial_state(batchsize=1, train=False)
    
    for i in six.moves.range(len(targets)):
        if targets[i] % offset_timing == 0:
            if i == 0:
                x_batch = mod.array([data[i] + [0,0]], dtype = 'float32')
            else:
                x_batch = mod.array([data[i] + [targets[i-1] % maze_size_x, targets[i-1] // maze_size_x]], dtype = 'float32')
        else:
            x_batch = mod.array([data[i] + [0, 0]], dtype = 'float32')
        t_batch = mod.array([targets[i]], dtype = 'int32')
        state, loss, accuracy = forward_one_step(x_batch, t_batch, state, train=False)
        sum_accuracy += accuracy.data
        if test == True:
            print('{} Target: ({}, {})'.format(accuracy.data, t_batch[0] % maze_size_x, 
                t_batch[0] // maze_size_x))

            # print('c: {}, h: {}'.format(state['c'].data, state['h'].data)) # show the hidden states
    return int(cuda.to_cpu(sum_accuracy))
                
# learning loop iterations
for loop in range(len(train_data_length)):

    # loop initialization
    whole_len = train_data_length[loop] 
    jump = whole_len // batchsize # = whole len
    cur_log_perp = mod.zeros(())
    start_at = time.time()
    cur_at = start_at
    epoch = 0
    accum_loss = chainer.Variable(mod.zeros((), dtype=np.float32))
    print('[train]')
    print('going to train {} iterations'.format(jump * n_epoch))
    
    # loop starts
    while epoch <= n_epoch:
        
        # initialize hidden state to 0
        state = make_initial_state()
        
        # train dataset
        if loop == 0:
            train_data, train_targets = generate_seq(whole_len, maze_size_x, maze_size_y)
        else:
            train_data, train_targets = generate_seq_remote(whole_len, maze_size_x, maze_size_y)
         
        for i in six.moves.range(jump):
        
            # forward propagation
            if train_targets[i] % offset_timing == 0:
                if i == 0:
                    x_batch = mod.array([train_data[i] + [0,0]], dtype = 'float32')
                else:
                    x_batch = mod.array([train_data[i] + [train_targets[i-1] % maze_size_x, train_targets[i-1] // maze_size_x]], dtype = 'float32')
            else:
                x_batch = mod.array([train_data[i] + [0, 0]], dtype = 'float32')
            t_batch = mod.array([train_targets[i]], dtype = 'int32')
            state, loss_i, acc_i = forward_one_step(x_batch, t_batch, state)
            accum_loss += loss_i
            cur_log_perp += loss_i.data.reshape(())
            
            # truncated BPTT
            if (i + 1) % bprop_len == 0:  
                optimizer.zero_grads()
                accum_loss.backward()
                accum_loss.unchain_backward()  # truncate
                accum_loss = chainer.Variable(mod.zeros((), dtype=np.float32))
                optimizer.clip_grads(grad_clip) # gradient clip
                optimizer.update()
            
            sys.stdout.flush()
        
        if (epoch + 1) % valid_len == 0: 
            
            # calculate accuracy, cumulative loss & throuput
            train_perp = evaluate(train_data, train_targets)
            valid_perp = evaluate(valid_data, valid_targets)
            perp = cuda.to_cpu(cur_log_perp) / valid_len
            now = time.time()
            throuput = valid_len / (now - cur_at)
            print('epoch {}: train perp: {:.2f} train classified {}/{}, valid classified {}/{} ({:.2f} epochs/sec)'
                .format(epoch+1, perp, train_perp, whole_len, valid_perp, len(valid_data), throuput))            
            cur_at = now
            
            #  termination criteria
            if perp < 0.001:
                break
            else:
                cur_log_perp.fill(0)   
            
        epoch += 1  
        
        # save the model    
        f = open('pretrained_model_'+str(maze_size_x)+'_'+str(maze_size_y)+'.pkl', 'wb')
        pickle.dump(model, f, 2)
        f.close()      
                    
    # Evaluate on test dataset
    print('[test]')
    test_perp = evaluate(test_data, test_targets, test=True)
    print('test classified: {}/{}'.format(test_perp, len(test_data)))
                
