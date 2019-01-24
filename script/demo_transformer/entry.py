# coding: utf-8

import pickle
import codecs
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from models.transformer import Transformer
from callbacks.scheduler import LRSchedulerPerBatch


class TokenTransformer(object):
    def __init__(self, src_dict, tar_dict, seq_len=None):
        self.src_dict = src_dict
        self.tar_dict = tar_dict
        self.seq_len = seq_len
        self.start_token = 2
        self.end_token = 3
        self.unknown_token = 1
        self.padding_token = 0

    def seq2tokens(self, sequence, mode="src"):
        tokens = [self.src_dict[w] if mode == "src" else self.tar_dict[w] for w in sequence]
        tokens = [2] + tokens + [3, 0]
        if self.seq_len is not None:
            res = np.zeros(self.seq_len, dtype=np.int32)
            res[:len(tokens)] = tokens
        else:
            res = np.array(tokens).reshape((1, -1))
        return res


if __name__ == "__main__":
    src_dict = pickle.load(codecs.open("english.dict", "rb"))
    tar_dict = pickle.load(codecs.open("german.dict", "rb"))

    token_model = TokenTransformer(src_dict, tar_dict)

    x_train = np.load("xtrain.npy")
    y_train = np.load("ytrain.npy")
    x_dev = np.load("xvalid.npy")
    y_dev = np.load("yvalid.npy")

    dim_size = 512
    trans = Transformer(src_dict=src_dict, tar_dict=tar_dict, model_dim=dim_size, num_layers=2, num_head=8,
                        length_limit=70, inner_dim=2048, dropout=0.1, use_pos_embedding=True)
    trans.compile(optimizer=Adam(0.001, 0.9, 0.98, epsilon=1e-9))
    trans.model.summary()

    model_saver = ModelCheckpoint("model/transformer.h5", save_weights_only=True, save_best_only=True)
    lr_scheduler = LRSchedulerPerBatch(dim_size=dim_size, warm_up=4000)

    task = "test"

    if task == "train":
        trans.model.fit([x_train, y_train], None, batch_size=64, epochs=30, validation_data=([x_dev, y_dev], None),
                        shuffle=True, verbose=1, callbacks=[lr_scheduler, model_saver])
    else:
        try:
            trans.model.load_weights("model/transformer.h5")
        except OSError:
            print("Training new model")

        # seq_tokens = token_model.seq2tokens("Two young , White males are outside near many bushes .".split())
        seq_tokens = token_model.seq2tokens("A black dog eats food .".split())
        decode_res = trans.decode(seq_tokens)
        decode_res1 = trans.decode_fast(seq_tokens)
        decode_res2 = trans.beam_search(seq_tokens, topk=5)
        print(decode_res)
        print(decode_res1)
        print(decode_res2)
