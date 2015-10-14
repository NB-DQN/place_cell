import chainer
import chainer.functions as F
import chainer.optimizers as Opt
from chainer import cuda
import numpy as np

class StackedAutoencoder:
    def __init__(self, gpu=-1):
        self.n_layer = 4

        # units
        number_of_units = [1080, 360, 120, 40, 12]

        # layers
        self.enc_layer = [
            F.Linear(number_of_units[0], number_of_units[1]),
            F.Linear(number_of_units[1], number_of_units[2]),
            F.Linear(number_of_units[2], number_of_units[3]),
            F.Linear(number_of_units[3], number_of_units[4])
        ]

        self.dec_layer = [
            F.Linear(number_of_units[4], number_of_units[3]),
            F.Linear(number_of_units[3], number_of_units[2]),
            F.Linear(number_of_units[2], number_of_units[1]),
            F.Linear(number_of_units[1], number_of_units[0])
        ]

        self.model = chainer.FunctionSet(
            enc1=self.enc_layer[0],
            enc2=self.enc_layer[1],
            enc3=self.enc_layer[2],
            enc4=self.enc_layer[3],
            dec1=self.dec_layer[0],
            dec2=self.dec_layer[1],
            dec3=self.dec_layer[2],
            dec4=self.dec_layer[3],
        )

        try:
            param = np.load('dae.param.npy.1')
            self.model.copy_parameters_from(param)
        except:
            pass

        if gpu >= 0:
            cuda.check_cuda_available()
            cuda.get_device(gpu).use()
            self.model.to_gpu()

    def encode(self, x):
        for l in range(0, self.n_layer):
            x = F.sigmoid(self.enc_layer[l](x))
        return x

    def decode(self, h):
        for l in range(0, self.n_layer):
            h = F.sigmoid(self.dec_layer[l](h))
        return h
