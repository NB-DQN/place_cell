import chainer
import chainer.functions as F
import chainer.optimizers as Opt
import numpy as np

from time import time

from dataset_generator import DatasetGenerator

## model definition

# units
number_of_units = [1080, 360, 120, 40, 12]

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

layerwise = [
    chainer.FunctionSet(enc=enc_layer[0], dec=dec_layer[3]),
    chainer.FunctionSet(enc=enc_layer[1], dec=dec_layer[2]),
    chainer.FunctionSet(enc=enc_layer[2], dec=dec_layer[1]),
    chainer.FunctionSet(enc=enc_layer[3], dec=dec_layer[0]),
]


def encode(x, layer):
    # if train:
    #     x = F.dropout(x, ratio=0.2)

    h = x
    for l in range(0, 4):
        if layer == l:
            return h
        h = F.sigmoid(enc_layer[l](h))
    return h

dg = DatasetGenerator((9, 9))
data = np.asarray(dg.generate_dataset_sae(5000), dtype='f')
N = len(data)

batchsize = 50
opt = Opt.Adam()

try:
    param = np.load('dae.param.npy')
    model.copy_parameters_from(param)
except:
    pass

flag = [False, False, False, False]

tstart = time()
for epoch in range(10000):
    tepoch = time()
    print('epoch : %d' % (epoch + 1))
    with open('dae.log', mode='a') as f:
        f.write("\n%d " % (epoch + 1))

    for l in range(0, 4):
        if flag[l]:
            continue

        opt.setup(layerwise[l])

        sum_err = 0.

        for i in range(0, N, batchsize):
            x_batch = np.asarray(data[i:i + batchsize])
            x = chainer.Variable(x_batch)
            targ = encode(x, l)
            enc = encode(x, l + 1)
            y = F.sigmoid(dec_layer[3 - l](enc))

            opt.zero_grads()
            err = F.mean_squared_error(y, targ)
            err.backward()
            opt.update()

            sum_err += float(err.data) * len(x_batch)

        sum_err /= N
        print("\t%d %f" % (l, sum_err))
        if sum_err < 0.001:
            flag[l] = True

    param = np.array(model.parameters)
    np.save('dae.param.npy', param)
    print("\tepoch time %d, total time %d" % (time() - tepoch, time() - tstart))
