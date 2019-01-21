# coding: utf-8

from keras import backend as K
from keras.callbacks import Callback


class LRSchedulerPerBatch(Callback):
    def __init__(self, dim_size, warm_up=4000):
        super(LRSchedulerPerBatch, self).__init__()
        self.basic = dim_size ** -0.5
        self.warm = warm_up ** -1.5
        self.step_num = 0

    def on_batch_begin(self, batch, logs=None):
        self.step_num += 1
        lr = self.basic * min(self.step_num ** -0.5, self.step_num * self.warm)
        K.set_value(self.model.optimizer.lr, lr)


class LRSchedulerPerEpoch(Callback):
    def __init__(self, dim_size, num_per_epoch, warm_up=4000):
        super(LRSchedulerPerEpoch, self).__init__()
        self.basic = dim_size ** -0.5
        self.warm = warm_up ** -1.5
        self.num_per_epoch = num_per_epoch
        self.step_num = 1

    def on_epoch_begin(self, epoch, logs=None):
        self.step_num += self.num_per_epoch
        r = self.basic * min(self.step_num ** -0.5, self.step_num * self.warm)
        K.set_value(self.model.optimizer.lr, lr)
