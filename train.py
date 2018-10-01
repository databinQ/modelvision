# coding: utf-8

import os


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

        if name is None:
            
