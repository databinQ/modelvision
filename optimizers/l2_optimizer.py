# coding: utf-8

from itertools import combinations
from keras import backend as K
from keras.optimizers import Optimizer, serialize


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

    @staticmethod
    def compute_l2_ratio(iters, l2_steps, l2_ratio):
        return K.sigmoid((iters - l2_steps / 2) * 8 / (l2_steps / 2)) * l2_ratio

    def add_decaying_l2_loss(self, loss, params, iters):
        l2_ratio = self.compute_l2_ratio(iters, self.l2_steps, self.l2_ratio)
        # Iterations for each weight in network to calculate regularization with l2
        for w in params:
            loss += K.sum(K.square(w) * l2_ratio)
        return loss

    def add_difference_l2_loss(self, loss, params):
        penalize_difference = {}
        for param in params:
            if hasattr(params, "penalize_difference"):
                param_name = param.penalize_difference
                if param_name not in penalize_difference:
                    penalize_difference[param_name] = []
                penalize_difference[param_name].append(param)

        for penalize_weights in penalize_difference.values():
            for w1, w2 in combinations(penalize_weights, 2):
                loss += K.sum(K.square(w1 - w2) * self.l2_difference_ratio)
        return loss

    def get_l2_loss(self, loss, params, iterations):
        """
        :param loss: current loss(samples related)
        :param params: all weights in the network
        :param iterations: training iterations
        :return: total loss with l2 regularization
        """
        i = K.cast(iterations, dtype=K.floatx())
        self.add_decaying_l2_loss(loss, params, i)
        self.add_difference_l2_loss(loss, params)
        return loss

    def get_updates(self, loss, params):
        loss = self.get_l2_loss(loss, params, self.optimizer.iterations)
        return self.optimizer.get_updates(loss, params)

    def get_config(self):
        config = {
            "optimizer": serialize(self.optimizer),  # Attention: use serialization
            "l2_steps": float(K.get_value(self.l2_steps)),
            "l2_ratio": float(K.get_value(self.l2_ratio)),
            "l2_difference_ratio": float(K.get_value(self.l2_difference_ratio)),
        }
        return config

    def get_weights(self):
        return self.optimizer.get_weights()

    def set_weights(self, weights):
        self.optimizer.set_weights(weights)
