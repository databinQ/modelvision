# coding: utf-8

import pickle
import numpy as np

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
model.fit([x_train, y_train], train_res, batch_size=64, epochs=5,
          validation_data=[[x_dev, y_dev], dev_res], shuffle=True, verbose=2)
