# coding: utf-8

from keras import backend as K
from keras.optimizers import Optimizer


class L2Optimizer(Optimizer):
    def __init__(self, optimizer, l2_steps=10000, l2_ratio=9e-5, l2_difference_ratio=1e-3):
        """Weight regularization, include:
        1. L2 regularization of all weights in the network, with exponential decaying ratio
        2. L2 regularization the difference between weights having same penalization name
        """
        super(L2Optimizer, self).__init__()

        self.optimizer = optimizer
        self.l2_steps = K.variable(l2_steps, dtype=K.floatx(), name="l2_steps")
        self.l2_ratio = K.variable(l2_ratio, dtype=K.floatx(), name="l2_ratio")
        self.l2_difference_ratio = K.variable(l2_difference_ratio, dtype=K.floatx(), name="l2_difference_ratio")

    def get_updates(self, loss, params):
        pass
