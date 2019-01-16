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
model.fit([x_train, y_train[:, :-1]], y_train[:, 1:], batch_size=64, epochs=5,
          validation_data=[[x_dev, y_dev[:, :-1]], y_dev[:, 1:]], shuffle=True)
