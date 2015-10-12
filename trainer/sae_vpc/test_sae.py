import chainer
import chainer.functions as F
import chainer.optimizers as Opt
import numpy as np
import matplotlib.pyplot as plt

from time import time

from dataset_generator import DatasetGenerator

## model definition

# units
number_of_units = [18000, 2000, 200, 40, 10]

# layers
enc_layer = [
    F.Linear(number_of_units[0], number_of_units[1]),
    F.Linear(number_of_units[1], number_of_units[2]),
    F.Linear(number_of_units[2], number_of_units[3]),
    F.Linear(number_of_units[3], number_of_units[4])
]

dec_layer = [
    F.Linear(number_of_units[4], number_of_units[3]),
    F.Linear(number_of_units[3], number_of_units[2]),
    F.Linear(number_of_units[2], number_of_units[1]),
    F.Linear(number_of_units[1], number_of_units[0])
]

model = chainer.FunctionSet(
    enc1=enc_layer[0],
    enc2=enc_layer[1],
    enc3=enc_layer[2],
    enc4=enc_layer[3],
    dec1=dec_layer[0],
    dec2=dec_layer[1],
    dec3=dec_layer[2],
    dec4=dec_layer[3],
)

param = np.load('dae.param.npy.1')
model.copy_parameters_from(param)

def encode(x):
    for l in range(0, 4):
        x = F.sigmoid(enc_layer[l](x))
    return x

def decode(h):
    for l in range(0, 4):
        h = F.sigmoid(dec_layer[l](h))
    return h

dg = DatasetGenerator((9, 9))
data = np.asarray(dg.generate_dataset_sae(10), dtype='f')
N = len(data)

for n in range(0, N):
    x = chainer.Variable(np.asarray([data[n]], dtype='f'))
    y = decode(encode(x))
    err = F.mean_squared_error(y, x)
    print(err.data)
    plt.subplot(2, 1, 1)
    plt.imshow(np.flipud(x.data.reshape((360, 50)).T), cmap=plt.cm.gray)
    plt.subplot(2, 1, 2)
    plt.imshow(np.flipud(y.data.reshape((360, 50)).T), cmap=plt.cm.gray)
    plt.show()

