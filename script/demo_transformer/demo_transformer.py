# coding: utf-8

import pickle
import numpy as np
# import tensorflow as tf
# import keras.backend.tensorflow_backend as ktf
#
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.333
# session = tf.Session(config=config)
# ktf.set_session(session)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

from models.transformer import Transformer

dic1 = pickle.load(open("dict1.pkl", "rb"))
dic2 = pickle.load(open("dict2.pkl", "rb"))

Xtrain, Ytrain = np.load("xtrain.npy"), np.load("ytrain.npy")
Xvalid, Yvalid = np.load("xvalid.npy"), np.load("yvalid.npy")

src_len = max(Xtrain.shape[1], Xvalid.shape[1])
tar_len = max(Ytrain.shape[1], Yvalid.shape[1])

x_train = np.zeros((Xtrain.shape[0], src_len), dtype=np.int32)
x_dev = np.zeros((Xvalid.shape[0], src_len), dtype=np.int32)
y_train = np.zeros((Ytrain.shape[0], tar_len), dtype=np.int32)
y_dev = np.zeros((Yvalid.shape[0], tar_len), dtype=np.int32)
x_train[:, :Xtrain.shape[1]] = Xtrain
x_dev[:, :Xvalid.shape[1]] = Xvalid
y_train[:, :Ytrain.shape[1]] = Ytrain
y_dev[:, :Yvalid.shape[1]] = Yvalid

model = Transformer(source_len=src_len, target_len=tar_len, scr_dict_size=len(dic1), tar_dict_size=len(dic2),
                    num_layers=1, use_pos_embedding=True, use_mask=True)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])

train_res = np.zeros((y_train.shape[0], y_train.shape[1] - 1, len(dic2)), dtype=np.int32)
dev_res = np.zeros((y_dev.shape[0], y_dev.shape[1] - 1, len(dic2)), dtype=np.int32)
for i in range(train_res.shape[0]):
    for j in range(train_res.shape[1]):
        train_res[i, j, y_train[i, j + 1]] = 1
for i in range(dev_res.shape[0]):
    for j in range(dev_res.shape[1]):
        dev_res[i, j, y_dev[i, j + 1]] = 1
model.summary()

batch_size = 64
num_train = len(x_train)
batches_per_epoch = (num_train + batch_size - 1) // batch_size


def batch_generator(X, Y, y):
    while 1:
        new_index = np.random.permutation(num_train)
        for i in range(batches_per_epoch):
            batch_index = new_index[i * batch_size: (i + 1) * batch_size]
            yield [X[batch_index, :], Y[batch_index, :]], y[batch_index]


model.fit_generator(generator=batch_generator(x_train, y_train, train_res),
                    steps_per_epoch=batches_per_epoch,
                    epochs=5,
                    verbose=2,
                    validation_data=[[x_dev, y_dev], dev_res],
                    use_multiprocessing=False)

# model.fit([x_train, y_train], train_res, batch_size=16, epochs=5,
#           validation_data=[[x_dev, y_dev], dev_res], shuffle=True, verbose=2)
