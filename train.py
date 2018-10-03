# coding: utf-8

import os
from collections import Iterable


class NLITaskTrain(object):
    def __init__(self, model, train_data, test_data, dev_data=None, optimizer=None, save_dir=None, name=None):
        self.model = model
        self.name = name

        """Data"""
        self.train_data = train_data
        self.test_data = test_data
        self.dev_data = dev_data

        """Train Methods"""
        self.optimizer = optimizer

        """Others"""
        self.save_dir = save_dir
        if self.save_dir is not None and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def train(self, batch_size=128, eval_interval=512, shuffle=True):
        return

    def train_multi_optimizer(self, batch_size=128, eval_interval=512, shuffle=True):
        assert isinstance(self.optimizer, Iterable) is True
        assert len(self.optimizer) > 1

        return
