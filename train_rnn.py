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

import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
                    
args = parser.parse_args()
mod = cuda if args.gpu >= 0 else np

# set parameters
n_epoch = 39   # number of epochs
n_units = 650  # number of units per layer
batchsize = 1   # minibatch size
bprop_len = 35   # length of truncated BPTT
grad_clip = 5    # gradient norm threshold to clip

# dataset
"""
train data, validation data & test data
train targets, validation targets & test targets

data: direction (0, 1, 2, 3)
target: one-hot vector that incidates the cordinate and novelty (dim: 1*80)
"""

train_data = 
valid_data = 
test_data = 
train_targets = 
valid_targets = 
test_targets = 

# model
model = FunctionSet(
    x_to_h = F.Linear(1, 50),
    h_to_h = F.Linear( 50, 50),
    h_to_y = F.Linear( 50, 81))

if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()
    
# optimizer
optimizer = optimizers.SGD()
optimizer.setup(model.collect_parameters())

# forward propagation
def forward_one_step(data, targets, state, train=True):
    """
    [warning]
    softmax_cross_entropy may not work because
    the function assumes one-hot y and single-number t while
    in this code both y and t are one-hot
    """
    if args.gpu >= 0:
        data = cuda.to_gpu(data)
        targets = cuda.to_gpu(targets)
    x = chainer.Variable(x, volatile=not train)
    t = chainer.Variable(t, volatile=not train)
    h_in = model.x_to_h(x) + model.h_to_h(state['h'])
    c, h = F.lstm(state['c'], h_in)
    y = model.h_to_y(h)
    state = {'c': c, 'h': h}
    return state, F.softmax_cross_entropy(y, t)

# initialize hidden states
def make_initial_state(batchsize=batchsize, train=True):
    return {name: chainer.Variable(mod.zeros((batchsize, n_units),
                                             dtype=np.float32),
                                   volatile=not train)
            for name in ('c', 'h')}
             
# evaluation routine
def evaluate(data, targets):
    sum_log_perp = mod.zeros(())
    state = make_initial_state(batchsize=1, train=False)
    for i in six.moves.range(dataset.size - 1):
        x_batch = data[i]
        t_batch = targets[i]
        state, loss = forward_one_step(x_batch, t_batch, state, train=False)
        sum_log_perp += loss.data.reshape(())

    return math.exp(cuda.to_cpu(sum_log_perp) / (dataset.size - 1)) 
                
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
    state, loss_i = forward_one_step(x_batch, t_batch, state)
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
        perp = evaluate(valid_data)
        print('epoch {} validation perplexity: {:.2f}'.format(epoch, perp))
        cur_at += time.time() - now  # skip time of evaluation

        if epoch >= 6:
            optimizer.lr /= 1.2
            print('learning rate =', optimizer.lr)

    sys.stdout.flush()

# Evaluate on test dataset
print('test')
test_perp = evaluate(test_data)
print('test perplexity:', test_perp)