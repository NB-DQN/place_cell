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

from dataset_generator import DatasetGenerator

# set parameters
n_epoch = 100000 # number of epochs
n_units = 60 # number of units per layer, len(train)=5 -> 20 might be the best
batchsize = 1 # minibatch size
bprop_len = 1 # length of truncated BPTT
valid_len = n_epoch // 100 # 1000 # epoch on which accuracy and perp are calculated
grad_clip = 5 # gradient norm threshold to clip
maze_size = (9, 9)

train_data_length = [20, 100]

# GPU
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
        help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
mod = cuda.cupy if args.gpu >= 0 else np

# generate dataset
dg = DatasetGenerator(maze_size)

# validation dataset
valid_data = dg.generate_seq(100)

# test dataset
test_data = dg.generate_seq(100)

# model
model = chainer.FunctionSet(
        x_to_h = F.Linear(64, n_units * 4),
        h_to_h = F.Linear(n_units, n_units * 4),
        h_to_y = F.Linear(n_units, 60))
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# optimizer
optimizer = optimizers.SGD(lr=1.)
optimizer.setup(model.collect_parameters())

# one-step forward propagation
def forward_one_step(x, t, state, train=True):
    # if args.gpu >= 0:
    #     data = cuda.to_gpu(data)
    #     targets = cuda.to_gpu(targets)
    x = chainer.Variable(x, volatile=not train)
    t = chainer.Variable(t, volatile=not train)
    h_in = model.x_to_h(x) + model.h_to_h(state['h'])
    c, h = F.lstm(state['c'], h_in)
    y = model.h_to_y(h)
    state = {'c': c, 'h': h}
    
    sigmoid_y = 1 / (1 + np.exp(-y.data))
    square_sum_error = ((t.data - sigmoid_y) ** 2).sum()

    return state, F.sigmoid_cross_entropy(y, t), square_sum_error

# initialize hidden state
def make_initial_state(batchsize=batchsize, train=True):
    return {name: chainer.Variable(mod.zeros((batchsize, n_units),
        dtype=np.float32),
        volatile=not train)
        for name in ('c', 'h')}

# evaluation
def evaluate(data, test=False):
    sum_error = 0
    state = make_initial_state(batchsize=1, train=False)

    for i in six.moves.range(len(data['input'])):
        x_batch = mod.asarray([data['input'][i]], dtype = 'float32')
        t_batch = mod.asarray([data['output'][i]], dtype = 'int32')
        state, loss, square_sum_error = forward_one_step(x_batch, t_batch, state, train=False)
        sum_error += square_sum_error
    return sum_error

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
        train_data = dg.generate_seq(whole_len)

        for i in six.moves.range(jump):

            # forward propagation
            x_batch = mod.array([train_data['input'][i]],  dtype = 'float32')
            t_batch = mod.array([train_data['output'][i]], dtype = 'int32')

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
            train_perp = evaluate(train_data)
            valid_perp = evaluate(valid_data)
            perp = cuda.to_cpu(cur_log_perp) / valid_len
            now = time.time()
            throuput = valid_len / (now - cur_at)
            print('epoch {}: train perp: {:.2f} train square-sum error: {:.2f}, valid square-sum error: {:.2f} ({:.2f} epochs/sec)'
                    .format(epoch+1, perp, train_perp, valid_perp, throuput))
            cur_at = now

            #  termination criteria
            if perp < 0.001:
                break
            else:
                cur_log_perp.fill(0)

        epoch += 1

        # save the model
        f = open('pretrained_model_'+str(maze_size[0])+'_'+str(maze_size[1])+'.pkl', 'wb')
        pickle.dump(model, f, 2)
        f.close()

    # Evaluate on test dataset
    print('[test]')
    test_square_sum_error = evaluate(test_data, test=True)
    print('test square-sum error: {:.2f}'.format(test_square_sum_error))

