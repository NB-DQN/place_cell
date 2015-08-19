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
n_epoch = 10   # number of epochs
n_units = 30  # number of units per layer
batchsize = 1   # minibatch size
bprop_len = 35   # length of truncated BPTT
grad_clip = 5    # gradient norm threshold to clip

# GPU
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')                    
args = parser.parse_args()
mod = cuda if args.gpu >= 0 else np

# data generation        
def generate_seq(seq_length, maze_size_x, maze_size_y):
    directions_2d = []
    locations_1d = [] # 1D coorinate

    current = (0, 0) # 2D coordinate
    for i in range(0, seq_length):
    	direction_choice = [[0.5, 0], [1, 0], [0, 0.5], [0, 1]]
        if current[0] == 0:
            direction_choice.remove([0.5, 0])
        if current[0] == maze_size_x-1:
            direction_choice.remove([1, 0])
        if current[1] == 0:
            direction_choice.remove([0, 0.5])
        if current[1] == maze_size_y-1:
            direction_choice.remove([0, 1])
        direction = random.choice(direction_choice)
        
        if   direction == [0.5, 0]:
            current = (current[0] - 1, current[1]    )
        elif direction == [1, 0]:
            current = (current[0] + 1, current[1]    )
        elif direction == [0, 0.5]:
            current = (current[0]    , current[1] - 1)
        elif direction == [0, 1]:
            current = (current[0]    , current[1] + 1)
        
        directions_2d.append(direction)
        locations_1d.append(current[0]+current[1]*maze_size_x)
        
    directions_2d = np.array(directions_2d, dtype='float32')
    locations_1d = np.array(locations_1d, dtype='int32')
    return directions_2d, locations_1d
    
# forward propagation
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

# initialize hidden states
def make_initial_state(batchsize=1, train=True):
    global mod
    return {name: chainer.Variable(mod.zeros((batchsize, n_units),
                                             dtype=np.float32),
                                   volatile=not train)
            for name in ('c', 'h')}
             
# evaluation routine
def evaluate(data, targets):
    sum_accuracy = 0
    state = make_initial_state(batchsize, train=False)
    for i in six.moves.range(targets.size):
        x_batch = np.array([data[i]])
        t_batch = np.array([targets[i]])z
        state, loss, accuracy = forward_one_step(x_batch, t_batch, state, train=False)
        sum_accuracy += accuracy.data
    return int(sum_accuracy)

def pretrain(maze_size_x=9, maze_size_y=9):
	
    # model
    global model
    model = chainer.FunctionSet(
        x_to_h = F.Linear(2, n_units*4),
        h_to_h = F.Linear( n_units, n_units*4),
        h_to_y = F.Linear( n_units, maze_size_x*maze_size_y))
    if args.gpu >= 0:
        cuda.init(args.gpu)
        model.to_gpu()    
    
    # dataset
    """
    input data: direction,  [x, y], 0.5: left/bottom move, 1: right/top move
    output target: coordinate (converted to 1D)
    """
    train_data, train_targets = generate_seq(1000,maze_size_x, maze_size_y)
    valid_data, valid_targets = generate_seq(100,maze_size_x, maze_size_y)
    test_data, test_targets = generate_seq(1000,maze_size_x, maze_size_y)
    
    # optimizer
    optimizer = optimizers.SGD()
    optimizer.setup(model.collect_parameters())
                    
    # learning loop
    whole_len = train_data.shape[0]
    jump = whole_len // batchsize
    cur_log_perp = mod.zeros(())
    epoch = 0
    start_at = time.time()
    cur_at = start_at
    state = make_initial_state()
    accum_loss = chainer.Variable(mod.zeros((), dtype=np.float32))
    print('going to train {} iterations'.format(jump * n_epoch))

    for i in six.moves.range(jump * n_epoch):
        x_batch = np.array([train_data[(jump * j + i) % whole_len]
                            for j in six.moves.range(batchsize)])
        t_batch = np.array([train_targets[(jump * j + i + 1) % whole_len]
                            for j in six.moves.range(batchsize)])
        state, loss_i, acc_i = forward_one_step(x_batch, t_batch, state)
        accum_loss += loss_i
        cur_log_perp += loss_i.data.reshape(())

        if (i + 1) % bprop_len == 0:  # Run truncated BPTT
            optimizer.zero_grads()
            accum_loss.backward()
            accum_loss.unchain_backward()  # truncate
            accum_loss = chainer.Variable(mod.zeros((), dtype=np.float32))

            optimizer.clip_grads(grad_clip)
            optimizer.update()

        if (i + 1) % 10000 == 0:
            now = time.time()
            throuput = 10000. / (now - cur_at)
            perp = math.exp(cuda.to_cpu(cur_log_perp) / 10000)
            print('iter {} training perplexity: {:.2f} ({:.2f} iters/sec)'.format(
                i + 1, perp, throuput))
            cur_at = now
            cur_log_perp.fill(0)

        if (i + 1) % jump == 0:
            epoch += 1
            print('evaluate')
            now = time.time()
            perp = evaluate(valid_data, valid_targets)
            print('epoch {} validation misclassified: {}'.format(epoch, perp))
            cur_at += time.time() - now  # skip time of evaluation
    
        sys.stdout.flush()
    
    # Evaluate on test dataset
    print('test')
    test_perp = evaluate(test_data, valid_targets)
    print('test misclassified:', test_perp)
    
    return model