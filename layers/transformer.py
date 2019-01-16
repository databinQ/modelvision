# coding: utf-8

from keras import backend as K
from keras.engine import Layer
from keras.layers import Conv1D


class PositionEmbedding(Layer):
    def __init__(self, size=None, mode="sum", **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.size = size
        self.mode = mode

    def call(self, inputs, **kwargs):
        if self.size is None or self.mode == "sum":
            self.size = int(inputs.shape[-1])

        position_j = 1.0 / K.pow(10000.0, 2 * K.arange(self.size / 2, dtype="float32") / self.size)
        position_j = K.expand_dims(position_j, 0)  # (1, dim/2)
        position_i = K.cumsum(K.ones_like(inputs[:, :, 0]), axis=1) - 1
        position_i = K.expand_dims(position_i, 2)  # (batch, seq_len, 1)
        position_ij = K.dot(position_i, position_j)  # (batch, seq_len, dim/2)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], axis=2)  # (batch, seq_len, dim)
        if self.mode == "sum":
            return inputs + position_ij
        elif self.mode == "concat":
            return K.concatenate([position_ij, inputs], axis=2)

    def compute_output_shape(self, input_shape):
        if self.mode == "sum":
            return input_shape
        elif self.mode == "concat":
            return tuple([input_shape[0], input_shape[1], input_shape[2] + self.size])


class MultiHeadAttention(Layer):
    def __init__(self, num_head, head_dim, model_dim, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_head = num_head
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.output_dim = self.num_head * self.head_dim

    def build(self, input_shape):
        self.Q_weight = self.add_weight(name="Q_weight", shape=(input_shape[0][-1], self.output_dim),
                                        initializer="glorot_uniform", trainable=True)
        self.K_weight = self.add_weight(name="K_weight", shape=(input_shape[1][-1], self.output_dim),
                                        initializer="glorot_uniform", trainable=True)
        self.V_weight = self.add_weight(name="V_weight", shape=(input_shape[2][-1], self.output_dim),
                                        initializer="glorot_uniform", trainable=True)
        self.O_weight = self.add_weight(name="O_weight", shape=(self.output_dim, self.model_dim),
                                        initializer="glorot_uniform", trainable=True)
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs, mask=None, seq_len=None):
        """
        Qs: (n, d_k)
        Ks: (m, d_k)
        Vs: (m, d_v)

        - Q and K have the same state dimension, so using dot to calculate the influence of each step in Q and K
            sequence.
        - The output sequence has n steps, each step is related to the steps in V. K and V have the same step number,
            therefore each step in output sequence is the sum of all step vectors in V, weighted with influence vector.
            Output matrix shape is (n, d_v)
        """
        Qs, Ks, Vs = inputs
        if seq_len is not None:
            Q_len, V_len = seq_len
        k_dim = K.shape(Qs)[-1]

        Qs = K.dot(Qs, self.Q_weight)  # (batch, n, model_dim)
        Qs = K.reshape(Qs, (-1, K.shape(Qs)[1], self.num_head, self.head_dim))  # (batch, n, n_head, head_dim)
        Qs = K.permute_dimensions(Qs, (0, 2, 1, 3))  # (batch, n_head, n, head_dim)
        Ks = K.dot(Ks, self.K_weight)  # (batch, m, model_dim)
        Ks = K.reshape(Ks, (-1, K.shape(Ks)[1], self.num_head, self.head_dim))  # (batch, m, n_head, head_dim)
        Ks = K.permute_dimensions(Ks, (0, 2, 1, 3))  # (batch, n_head, m, head_dim)
        Vs = K.dot(Vs, self.V_weight)  # (batch, m, model_dim)
        Vs = K.reshape(Vs, (-1, K.shape(Vs)[1], self.num_head, self.head_dim))  # (batch, m, n_head, head_dim)
        Vs = K.permute_dimensions(Vs, (0, 2, 1, 3))  # (batch, n_head, m, head_dim)

        # compute dot
        A = K.batch_dot(Qs, Ks, axes=[3, 3]) / self.model_dim  # (batch, n_head, n, m)
        A = K.permute_dimensions(A, (0, 3, 2, 1))  # (batch, m, n, n_head)
        if seq_len is not None:
            A = self.conduct_mask(A, V_len, method="add")
        A = K.permute_dimensions(A, (0, 3, 2, 1))  # (batch, n_head, n, m)
        A = K.softmax(A)

        # compute output
        output = K.batch_dot(A, Vs, axes=[3, 2])  # (batch, n_head, n, head_dim)
        output = K.permute_dimensions(output, (0, 2, 1, 3))  # (batch, n, n_head, head_dim)
        output = K.reshape(output, shape=(-1, K.shape(output)[1], self.output_dim))  # (batch, n, n_head * head_dim)
        if seq_len is not None:
            output = self.conduct_mask(output, Q_len, method="mul")
        output = K.dot(output, self.O_weight)  # (batch, n, model_dim)
        return output

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0][0], input_shape[0][1], self.output_dim])

    @staticmethod
    def conduct_mask(inputs, seq_len, method="mul"):
        # seq_len: (batch,)
        mask = K.one_hot(seq_len, K.shape(inputs)[1])  # (batch, seq_len)
        mask = 1 - K.cumsum(mask, axis=1)
        for _ in range(len(inputs.shape) - 2):
            mask = K.expand_dims(mask, -1)

        if method == "add":
            return inputs - (1 - mask) * 1e12
        elif method == "mul":
            return inputs * mask


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.gamma = self.add_weight(name="gamma", shape=(input_shape[-1],), initializer="ones", trainable=True)
        self.beta = self.add_weight(name="beta", shape=(input_shape[-1],), initializer="zeros", trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        std = K.std(inputs, axis=-1, keepdims=True)
        return self.gamma * (inputs - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class PositionwiseFeedForward(Layer):
    def __init__(self, inner_dim, **kwargs):
        super(PositionwiseFeedForward, self).__init__(**kwargs)
        self.inner_dim = inner_dim

    def call(self, inputs, **kwargs):
        inner_output = Conv1D(self.inner_dim, 1, activation="relu")(inputs)
        output = Conv1D(int(inputs.shape[-1]), 1)(inner_output)
        return output


class EncoderUnit(Layer):
    def __init__(self, num_head, head_dim, model_dim, inner_dim, **kwargs):
        super(EncoderUnit, self).__init__(**kwargs)
        self.num_head = num_head
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.inner_dim = inner_dim

    def call(self, inputs, seq_len=None):
        if seq_len is not None:
            seq_lens = [seq_len, seq_len]
        attn_output = MultiHeadAttention(num_head=self.num_head, head_dim=self.head_dim, model_dim=self.model_dim)(
            [inputs, inputs, inputs], seq_len=seq_lens)
        output1 = LayerNormalization()(attn_output + inputs)
        feed_output = PositionwiseFeedForward(inner_dim=self.inner_dim)(output1)
        output2 = LayerNormalization()(feed_output + output1)
        return output2


class DecoderUnit(Layer):
    def __init__(self, num_head, head_dim, model_dim, inner_dim, **kwargs):
        super(DecoderUnit, self).__init__(**kwargs)
        self.num_head = num_head
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.inner_dim = inner_dim

    def call(self, inputs, input_seq_len=None, output_seq_len=None):
        encode_inputs, output_inputs = inputs
        if input_seq_len is None or output_seq_len is None:
            seq_lens1, seq_lens2 = None, None
        else:
            seq_lens1 = [output_seq_len, output_seq_len]
            seq_lens2 = [output_seq_len, input_seq_len]
        output_attn_output = MultiHeadAttention(num_head=self.num_head, head_dim=self.head_dim,
                                                model_dim=self.model_dim)([output_inputs, output_inputs, output_inputs],
                                                                          seq_len=seq_lens1)
        output1 = LayerNormalization()(output_attn_output + output_inputs)
        middle_attn_output = MultiHeadAttention(num_head=self.num_head, head_dim=self.head_dim,
                                                model_dim=self.model_dim)([output1, encode_inputs, encode_inputs],
                                                                          seq_len=seq_lens2)
        output2 = LayerNormalization()(middle_attn_output + output1)
        feed_output = PositionwiseFeedForward(inner_dim=self.inner_dim)(output1)
        output3 = LayerNormalization()(feed_output + output2)
        return output3


class Encode(Layer):
    def __init__(self, num_layers, num_head, head_dim, model_dim, inner_dim, **kwargs):
        super(Encode, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.num_head = num_head
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.inner_dim = inner_dim

    def call(self, inputs, seq_len=None):
        for _ in range(self.num_layers):
            inputs = EncoderUnit(num_head=self.num_head, head_dim=self.head_dim, model_dim=self.model_dim,
                                 inner_dim=self.inner_dim)(inputs, seq_len=seq_len)
        return inputs


class Decode(Layer):
    def __init__(self, num_layers, num_head, head_dim, model_dim, inner_dim, **kwargs):
        super(Decode, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.num_head = num_head
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.inner_dim = inner_dim

    def call(self, inputs, input_seq_len=None, output_seq_len=None):
        outputs, encoder_outputs = inputs
        for _ in range(self.num_layers):
            outputs = DecoderUnit(num_head=self.num_head, head_dim=self.head_dim, model_dim=self.model_dim,
                                  inner_dim=self.inner_dim)([encoder_outputs, outputs], input_seq_len=input_seq_len,
                                                            output_seq_len=output_seq_len)
        return outputs
