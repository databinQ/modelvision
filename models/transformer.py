# coding: utf-8

import tensorflow as tf
from keras.layers import *
from keras import backend as K
from keras.engine import Model

from layers.transformer import *


def get_mask_seq2seq(q, k):
    """0 means masked"""
    ones = K.expand_dims(K.ones_like(q, dtype="float32"), -1)  # (batch, q, 1)
    mask = K.expand_dims(K.cast(K.not_equal(k, 0), dtype="float32"), 1)  # (batch, 1, k)
    return K.batch_dot(ones, mask, axes=[2, 1])  # (batch, q, k)


def get_mask_self(s):
    return K.cumsum(tf.eye(tf.shape(s)[1], batch_shape=tf.shape(s)[:1]), axis=1)


class Transformer(object):
    def __init__(self, src_dict, tar_dict=None, length_limit=70,
                 num_layers=6, model_dim=512, num_head=8, head_dim=None, inner_dim=2048, dropout=0.1,
                 use_pos_embedding=True, share_embedding=False, inputs=None, outputs=None, name="Transformer"):
        if inputs is not None and outputs is not None:
            super(Transformer, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        self.src_dict = src_dict
        self.tar_dict = tar_dict if tar_dict is not None else self.src_dict
        self.src_token_dict = {v: k for k, v in self.src_dict.items()}
        self.tar_token_dict = {v: k for k, v in self.tar_dict.items()}
        self.scr_dict_size = len(self.src_dict)
        self.tar_dict_size = len(self.tar_dict) if tar_dict is not None else self.scr_dict_size

        self.length_limit = length_limit

        self.num_layers = num_layers
        self.num_head = num_head
        self.model_dim = model_dim
        self.head_dim = head_dim if head_dim is not None else int(model_dim / num_head)
        self.inner_dim = inner_dim

        self.dropout = dropout
        self.use_pos_embedding = use_pos_embedding
        self.share_embedding = share_embedding

        self.source_embedding = None
        self.target_embedding = None
        self.position_embedding = None
        self.encoder = None
        self.decoder = None
        self.softmax = None
        self.model = None
        self.output_model = None

        self.decode_build = False
        self.encoder_model = None
        self.decoder_model = None

    def compile(self, optimizer="adam"):
        source_input = Input(shape=(None,), dtype="int32")
        target_input = Input(shape=(None,), dtype="int32")

        target_decode_in = Lambda(
            lambda x: K.slice(x, start=[0, 0], size=[K.shape(target_input)[0], K.shape(target_input)[1] - 1])
        )(target_input)
        target_decode_out = Lambda(
            lambda x: K.slice(x, start=[0, 1], size=[K.shape(target_input)[0], K.shape(target_input)[1] - 1])
        )(target_input)

        src_mask = Lambda(lambda x: get_mask_seq2seq(x, x))(source_input)
        tar_mask = Lambda(lambda x: self.get_self_mask(x))(target_decode_in)
        encode_mask = Lambda(lambda x: get_mask_seq2seq(x[0], x[1]))([target_decode_in, source_input])

        self.source_embedding = Embedding(input_dim=self.scr_dict_size, output_dim=self.model_dim)
        if self.share_embedding:
            self.target_embedding = self.source_embedding
        else:
            self.target_embedding = Embedding(input_dim=self.tar_dict_size, output_dim=self.model_dim)

        if self.use_pos_embedding:
            self.position_embedding = PositionEmbedding(mode="sum")

        src_x = self.source_embedding(source_input)
        if self.use_pos_embedding:
            src_x = self.position_embedding(src_x)

        src_x = Dropout(self.dropout)(src_x)

        self.encoder = Encode(num_layers=self.num_layers, num_head=self.num_head, head_dim=self.head_dim,
                              model_dim=self.model_dim, inner_dim=self.inner_dim, dropout=self.dropout)
        encoder_output = self.encoder(src_x, masks=src_mask)

        tar_x = self.target_embedding(target_decode_in)
        if self.use_pos_embedding:
            tar_x = self.position_embedding(tar_x)

        self.decoder = Decode(num_layers=self.num_layers, num_head=self.num_head, head_dim=self.head_dim,
                              model_dim=self.model_dim, inner_dim=self.inner_dim, dropout=self.dropout)
        decoder_output = self.decoder([tar_x, encoder_output], self_mask=tar_mask, encode_mask=encode_mask)

        self.softmax = TimeDistributed(Dense(self.tar_dict_size))

        output = self.softmax(decoder_output)

        loss = Lambda(lambda x: self._get_loss(*x))([output, target_decode_out])

        self.model = Model([source_input, target_input], loss)
        self.model.add_loss([loss])
        self.model.compile(optimizer, None)

        self.model.metrics_names.append("ppl")
        self.model.metrics_tensors.append(Lambda(K.exp)(loss))
        self.model.metrics_names.append("accuracy")
        self.model.metrics_tensors.append(Lambda(lambda x: self._get_acc(x[0], x[1]))([output, target_decode_out]))

        self.output_model = Model([source_input, target_input], output)

    @staticmethod
    def get_encode_mask(src_seq):
        return get_mask_seq2seq(src_seq, src_seq)

    @staticmethod
    def get_self_mask(tar_seq):
        self_mask1 = get_mask_seq2seq(tar_seq, tar_seq)
        self_mask2 = get_mask_self(tar_seq)
        return K.minimum(self_mask1, self_mask2)

    @staticmethod
    def _get_loss(y_pred, y_true):
        y_true = tf.cast(y_true, dtype="int32")
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        mask = tf.cast(tf.not_equal(y_true, 0), dtype="float32")
        loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
        return tf.reduce_mean(loss)

    @staticmethod
    def _get_acc(y_pred, y_true):
        mask = tf.cast(tf.not_equal(y_true, 0), dtype="float32")
        corr = K.cast(K.equal(K.cast(y_true, dtype="int32"), K.cast(K.argmax(y_pred, -1), dtype="int32")),
                      dtype="float32")
        acc = K.sum(corr * mask, -1) / K.sum(mask, -1)
        return K.mean(acc)

    def decode_fast(self, seq):
        decode_tokens = []
        target_seq = np.zeros(shape=(1, self.length_limit), dtype=np.int32)
        target_seq[0, 0] = 2
        for i in range(self.length_limit - 1):
            output = self.output_model.predict_on_batch([seq, target_seq])
            max_prob_index = np.argmax(output[0, i, :])
            max_prob_token = self.tar_token_dict[max_prob_index]
            decode_tokens.append(max_prob_token)
            if max_prob_index == 3:
                break
            target_seq[0, i + 1] = max_prob_index
        return " ".join(decode_tokens)

    def _build_encoder(self):
        source_input = Input(shape=(None,), dtype="int32")

        src_mask = Lambda(lambda x: get_mask_seq2seq(x, x))(source_input)

        src_x = self.source_embedding(source_input)
        if self.use_pos_embedding:
            src_x = self.position_embedding(src_x)

        encoder_output = self.encoder(src_x, masks=src_mask)
        self.encoder_model = Model([source_input], encoder_output)
        self.encoder_model.compile('adam', 'mse')

    def _build_decoder(self):
        source_input = Input(shape=(None,), dtype="int32")
        target_input = Input(shape=(None,), dtype="int32")
        encoder_output = Input(shape=(None, self.model_dim))

        tar_mask = Lambda(lambda x: self.get_self_mask(x))(target_input)
        encode_mask = Lambda(lambda x: get_mask_seq2seq(x[0], x[1]))([target_input, source_input])

        tar_x = self.target_embedding(target_input)
        if self.use_pos_embedding:
            tar_x = self.position_embedding(tar_x)

        decoder_output = self.decoder([tar_x, encoder_output], self_mask=tar_mask, encode_mask=encode_mask)
        final_output = self.softmax(decoder_output)
        self.decoder_model = Model([source_input, target_input, encoder_output], final_output)
        self.decoder_model.compile('adam', 'mse')

    def _build_decode_model(self):
        self._build_encoder()
        self._build_decoder()
        self.decode_build = True

    def decode(self, seq):
        if not self.decode_build:
            self._build_decode_model()

        decode_tokens = []
        target_seq = np.zeros(shape=(1, self.length_limit), dtype=np.int32)
        target_seq[0, 0] = 2

        encoder_output = self.encoder_model.predict_on_batch([seq])
        for i in range(self.length_limit - 1):
            output = self.decoder_model.predict_on_batch([seq, target_seq, encoder_output])
            max_prob_index = np.argmax(output[0, i, :])
            max_prob_token = self.tar_token_dict[max_prob_index]
            decode_tokens.append(max_prob_token)
            if max_prob_index == 3:
                break
            target_seq[0, i + 1] = max_prob_index
        return " ".join(decode_tokens)

    def beam_search(self, seq, topk=3):
        if not self.decode_build:
            self._build_decode_model()

        seq = np.repeat(seq, topk, axis=0)
        encoder_output = self.encoder_model.predict_on_batch([seq])

        final_results = []
        topk_prob = np.zeros((topk,), dtype=np.float32)
        decode_tokens = [[] for _ in range(topk)]

        target_seq = np.zeros((topk, self.length_limit), dtype=np.int32)
        target_seq[:, 0] = 2

        last_k = 1

        for i in range(self.length_limit - 1):
            if last_k == 0 or len(final_results) > topk * 3:
                break  # stop conditions

            target_output = self.decoder_model.predict_on_batch([seq, target_seq, encoder_output])
            output = np.exp(target_output[:, i, :])
            output = output / np.sum(output, axis=-1, keepdims=True)
            output = np.log(output + 1e-8)  # use `log` transformation to avoid tiny probability

            candidates = []

            for k, probs in zip(range(last_k), output):
                if target_seq[k, i] == 3:
                    continue

                word_p_sort = sorted(list(enumerate(probs)), key=lambda x: x[1], reverse=True)
                for ind, wp in word_p_sort[:topk]:
                    candidates.append((k, ind, topk_prob[k] + wp))

            candidates = sorted(candidates,key=lambda x: x[-1], reverse=True)
            candidates = candidates[:topk]

            target_seq_bk = target_seq.copy()

            for new_k, cand in enumerate(candidates):
                k, ind, seq_p = cand
                target_seq[new_k] = target_seq_bk[k]
                target_seq[new_k, i + 1] = ind
                topk_prob[new_k] = seq_p
                decode_tokens.append(decode_tokens[k] + [self.tar_token_dict[ind]])
                if ind == 3:
                    final_results.append((decode_tokens[k], seq_p))

            decode_tokens = decode_tokens[topk:]
            last_k = len(decode_tokens)

        final_results = [(x, y / (len(x) + 1)) for x, y in final_results]
        final_results = sorted(final_results, key=lambda x: x[1], reverse=True)
        return final_results
