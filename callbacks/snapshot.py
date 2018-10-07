# coding: utf-8

import os
import numpy as np
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler


class SnapshotCallback(Callback):
    """Take snapshot, in other words saving model while optimizing"""

    def __init__(self, epoch_num, model_num, model_path=None, prefix="Model"):
        super(SnapshotCallback, self).__init__()
        self.iter_length = epoch_num // model_num
        self.prefix = prefix
        self.model_path = model_path

    def on_epoch_end(self, epoch, logs=None):
        """Save model at the end of each epoch, only the final model of each iteration will be saved"""
        if epoch != 0 and (epoch + 1) % self.iter_length == 0:
            model_path = self.model_path + self.prefix + "-%d.h5".format((epoch + 1) // self.iter_length)
            self.model.save_weights(model_path, overwrite=True)


class SnapshotCallbackBuilder(object):
    def __init__(self, T, M, init_lr=0.1):
        self.T = T
        self.M = M
        self.alpha_0 = init_lr

    def get_callbacks(self, model_path="./", snapshot_prefix="Model", best_prefix="Model-best"):
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        callback_list = [
            ModelCheckpoint("%s.h5".format(best_prefix), monitor="val_loss", save_best_only=True, save_weights_only=True),
            LearningRateScheduler(self._cosine_anneal_schedule),
            SnapshotCallback(self.T, self.M, model_path=model_path, prefix=snapshot_prefix)
        ]
        return callback_list

    def _cosine_anneal_schedule(self, t):
        t = np.pi * t % (self.T // self.M)
        t = t / (self.T // self.M)
        return self.alpha_0 / 2.0 * (np.cos(t) + 1)
