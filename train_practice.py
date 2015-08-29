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

import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

# set parameters
n_epoch = 500000 # number of epochs
n_units = 25 # number of units per layer, len(train)=5->20
batchsize = 1 # minibatch size
bprop_len = 1 # length of truncated BPTT
print_len = n_epoch // 100 # print results
grad_clip = 5 # gradient norm threshold to clip
maze_size_x = 9
maze_size_y = 9

# GPU
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')                    
args = parser.parse_args()
mod = cuda if args.gpu >= 0 else np

# generate dataset
"""
input data: direction (2D)
output target: coordinate (converted to 1D)
"""
def generate_seq(seq_length, maze_size_x, maze_size_y):
    directions_2d = [] # [x, y], 0.5: left/bottom move, 1: right/top move
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
        
        directions_2d.append(direction)
        locations_1d.append(current[0]+current[1]*maze_size_x)
        
    directions_2d = np.array(directions_2d, dtype='float32')
    locations_1d = np.array(locations_1d, dtype='int32')
    return directions_2d, locations_1d

# validation dataset
valid_data, valid_targets = generate_seq(20, maze_size_x, maze_size_y)

# test dataset
test_data, test_targets = generate_seq(100, maze_size_x, maze_size_y)

# model
model = chainer.FunctionSet(
    x_to_h = F.Linear(2, n_units*4),
    h_to_h = F.Linear( n_units, n_units*4),
    h_to_y = F.Linear( n_units, 81))
if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()
    
# optimizer
optimizer = optimizers.SGD(lr=1.)
optimizer.setup(model.collect_parameters())

# one-step forward propagation
def forward_one_step(data, targets, state, train=True):
    if args.gpu >= 0:
        data = cuda.to_gpu(data)
        targets = cuda.to_gpu(targets)
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
    sum_accuracy = 0
    state = make_initial_state(batchsize=1, train=False)
    for i in six.moves.range(targets.size):
        x_batch = np.array([data[i]])
        t_batch = np.array([targets[i]])
        state, loss, accuracy = forward_one_step(x_batch, t_batch, state, train=False)
        sum_accuracy += accuracy.data
        if test == True:
            print('{} Target: ({}, {})'.format(accuracy.data, t_batch[0] % maze_size_x, 
                t_batch[0] // maze_size_x))
    return int(cuda.to_cpu(sum_accuracy))
                
# learning loop iterations
train_data_length = [5, 10, 20]
for loop in range(len(train_data_length)):

    # loop initialization
    whole_len = train_data_length[loop] 
    jump = whole_len // batchsize # = whole len
    cur_log_perp = mod.zeros(())
    start_at = time.time()
    cur_at = start_at
    epoch = 0
    accum_loss = chainer.Variable(mod.zeros((), dtype=np.float32))
    print('going to train {} iterations'.format(jump * n_epoch))
    
    # loop starts
    while epoch <= n_epoch:
        
        # initialize hidden state to 0
        state = make_initial_state()
        
        # train dataset
        train_data, train_targets = generate_seq(whole_len, maze_size_x, maze_size_y)
        
        for i in six.moves.range(jump):
        
            # forward propagation
            x_batch = np.array([train_data[i % whole_len]])
            t_batch = np.array([train_targets[i % whole_len]])        
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
        
        if (epoch + 1) % print_len == 0: 
            
            # print accuracy, cumulative loss & throuput
            train_perp = evaluate(train_data, train_targets)
            valid_perp = evaluate(valid_data, valid_targets)
            perp = math.exp(cuda.to_cpu(cur_log_perp) / print_len)
            now = time.time()
            throuput = print_len / (now - cur_at)
            print('epoch {}: train perp: {:.2f} train classified {}/{}, valid classified {}/{} ({:.2f} epochs/sec)'
                .format(epoch+1, perp, train_perp, whole_len, valid_perp, valid_data.shape[0], throuput))            
            cur_at = now
            
            #  termination criteria
            if perp < 1.01:
                break
            else:
                cur_log_perp.fill(0)   
            
        epoch += 1      
                    
    # Evaluate on test dataset
    print('test')
    test_perp = evaluate(test_data, test_targets, test=True)
    print('test classified:', test_perp)
    