# coding: utf-8

import os
import numpy as np
from collections import Iterable
from utils.logger import base_logger as logger
from keras.callbacks import TensorBoard


class NLITaskTrain(object):
    def __init__(self, model, train_data, test_data, dev_data=None, optimizer=None, log_dir=None, save_dir=None,
                 name=None):
        self.model = model
        self.name = name

        """Data"""
        self.train_label = train_data[-1]
        self.train_data = train_data[:-1]
        self.test_data = test_data
        self.dev_data = dev_data
        if self.dev_data is not None:
            self.dev_label = self.dev_data[-1]
            self.dev_data = self.dev_data[:-1]

        """Train Methods"""
        self.optimizer = optimizer
        self.current_optimizer = None
        self.current_optimizer_id = -1
        self.current_switch_steps = -1

        """Others"""
        self.log_dir = log_dir
        if self.log_dir is not None and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.logger = TensorBoard(log_dir=self.log_dir)

        self.save_dir = save_dir
        if self.save_dir is not None and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def train(self, batch_size=128, eval_interval=512, shuffle=True):
        return

    def train_multi_optimizer(self, batch_size=128, eval_interval=512, shuffle=True):
        assert isinstance(self.optimizer, Iterable) is True
        assert len(self.optimizer) > 1

        self.current_optimizer = None
        self.current_optimizer_id = -1
        self.current_switch_steps = -1

        self.init_optimizer()
        self.model.summary()

        train_steps, no_progress_steps, epoch = 0, 0, 0
        train_batch_start = 0
        best_loss = np.inf

        while True:
            if shuffle:
                random_index = np.random.permutation(len(self.train_label))
                self.train_data = [data[random_index] for data in self.train_data]
                self.train_label = self.train_label[random_index]

            dev_loss, dev_acc = self.evaluate(batch_size=batch_size)
            self.logger.on_epoch_end(epoch=epoch, logs={"dev_loss": dev_loss, "dev_acc": dev_acc})
            self.model.save(self.save_dir + "epoch{}-loss{}-acc{}.model".format(epoch, dev_loss, dev_acc))
            epoch += 1
            no_progress_steps += 1

            if dev_loss < best_loss:
                best_loss = dev_loss
                no_progress_steps = 0

            if no_progress_steps > self.current_switch_steps:
                self.switch_optimizer()
                no_progress_steps = 0

            for i in range(eval_interval):
                train_loss, train_acc = self.model.train_on_batch(
                    [data[train_batch_start: train_batch_start + batch_size] for data in self.train_data],
                    self.train_label[train_batch_start: train_batch_start + batch_size]
                )
                self.logger.on_batch_end(train_steps, logs={"train_loss": train_loss, "train_acc": train_acc})

                train_steps += 1
                train_batch_start += batch_size
                if train_batch_start > len(self.train_label):
                    train_batch_start = 0
                    if shuffle:
                        random_index = np.random.permutation(len(self.train_label))
                        self.train_data = [data[random_index] for data in self.train_data]
                        self.train_label = self.train_label[random_index]

    def init_optimizer(self):
        self.current_optimizer_id = 0
        self.current_optimizer, self.current_switch_steps = self.optimizer[self.current_optimizer_id]
        self.model.compile(optimizer=self.current_optimizer,
                           loss="binary_crossentropy",
                           metrics=["acc"])
        self.logger.set_model(self.model)
        logger.info("Switch to {} optimizer".format(self.current_optimizer))

    def evaluate(self, X=None, y=None, batch_size=None):
        if X is None:
            X, y = self.dev_data, self.dev_label

        loss, acc = self.model.evaluate(X, y, batch_size=batch_size)
        return loss, acc

    def switch_optimizer(self):
        self.current_optimizer_id += 1
        if self.current_optimizer_id >= len(self.optimizer):
            logger.info("Training processes finished")
            exit(0)

        self.current_optimizer, self.current_switch_steps = self.optimizer[self.current_optimizer_id]
        self.model.compile(optimizer=self.current_optimizer,
                           loss="binary_crossentropy",
                           metrics=["acc"])
        self.logger.set_model(self.model)
        logger.info("Switch to {} optimizer".format(self.current_optimizer))


