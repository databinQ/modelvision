# coding: utf-8

from keras import backend as K
from keras.engine import Layer
from keras.layers import Conv1D, Dropout, Add


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
    def __init__(self, num_head, head_dim, model_dim, dropout, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_head = num_head
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.output_dim = self.num_head * self.head_dim
        self.dropout = dropout

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

    def call(self, inputs, masks=None):
        """
        Qs: (n, d_k)
        Ks: (m, d_k)
        Vs: (m, d_v)
        masks: (n, m)

        - Q and K have the same state dimension, so using dot to calculate the influence of each step in Q and K
            sequence.
        - The output sequence has n steps, each step is related to the steps in V. K and V have the same step number,
            therefore each step in output sequence is the sum of all step vectors in V, weighted with influence vector.
            Output matrix shape is (n, d_v)
        """
        Qs, Ks, Vs = inputs
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
        A = K.permute_dimensions(A, (0, 2, 3, 1))  # (batch, n, m, n_head)
        if masks is not None:
            A = self.conduct_mask(A, masks, method="add")
        A = K.permute_dimensions(A, (0, 3, 1, 2))  # (batch, n_head, n, m)
        A = K.softmax(A)
        A = Dropout(self.dropout)(A)

        # compute output
        output = K.batch_dot(A, Vs, axes=[3, 2])  # (batch, n_head, n, head_dim)
        output = K.permute_dimensions(output, (0, 2, 1, 3))  # (batch, n, n_head, head_dim)
        output = K.reshape(output, shape=(-1, K.shape(output)[1], self.output_dim))  # (batch, n, n_head * head_dim)
        # if masks is not None:
        #     output = self.conduct_mask(output, masks, method="mul")
        output = K.dot(output, self.O_weight)  # (batch, n, model_dim)
        return output

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0][0], input_shape[0][1], self.output_dim])

    @staticmethod
    def conduct_mask(inputs, masks, method="mul"):
        """
        inputs: (batch, m, n, n_head)
        masks: (batch, m, n)
        """
        mask = K.expand_dims(masks, -1)  # (batch, m, n, 1)

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


class PositionWiseFeedForward(object):
    def __init__(self, inner_dim, model_dim):
        self.inner_dim = inner_dim
        self.model_dim = model_dim
        self.conv1 = Conv1D(self.inner_dim, 1, activation="relu")
        self.conv2 = Conv1D(self.model_dim, 1)

    def __call__(self, inputs):
        inner_output = self.conv1(inputs)
        output = self.conv2(inner_output)
        return output


class EncoderUnit(object):
    def __init__(self, num_head, head_dim, model_dim, inner_dim, dropout):
        self.num_head = num_head
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.inner_dim = inner_dim
        self.dropout = dropout
        self.multi_head = MultiHeadAttention(num_head=self.num_head, head_dim=self.head_dim, model_dim=self.model_dim,
                                             dropout=self.dropout)
        self.multi_head_dropout = Dropout(self.dropout)
        self.multi_head_norm = LayerNormalization()
        self.point_wise = PositionWiseFeedForward(inner_dim=self.inner_dim, model_dim=self.model_dim)
        self.point_wise_dropout = Dropout(self.dropout)
        self.point_wise_norm = LayerNormalization()

    def __call__(self, inputs, masks=None):
        attn_output = self.multi_head([inputs, inputs, inputs], masks=masks)
        attn_output = self.multi_head_dropout(attn_output)
        output1 = Add()([attn_output, inputs])
        output1 = self.multi_head_norm(output1)
        feed_output = self.point_wise(output1)
        feed_output = self.point_wise_dropout(feed_output)
        output2 = Add()([feed_output, output1])
        output2 = self.point_wise_norm(output2)
        return output2


class DecoderUnit(object):
    def __init__(self, num_head, head_dim, model_dim, inner_dim, dropout):
        self.num_head = num_head
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.inner_dim = inner_dim
        self.dropout = dropout
        self.decode_multi_head = MultiHeadAttention(num_head=self.num_head, head_dim=self.head_dim,
                                                    model_dim=self.model_dim, dropout=self.dropout)
        self.decode_dropout = Dropout(self.dropout)
        self.decode_norm = LayerNormalization()
        self.encode_multi_head = MultiHeadAttention(num_head=self.num_head, head_dim=self.head_dim,
                                                    model_dim=self.model_dim, dropout=self.dropout)
        self.encode_dropout = Dropout(self.dropout)
        self.encode_norm = LayerNormalization()
        self.point_wise = PositionWiseFeedForward(inner_dim=self.inner_dim, model_dim=self.model_dim)
        self.point_wise_dropout = Dropout(self.dropout)
        self.point_wise_norm = LayerNormalization()

    def __call__(self, inputs, self_mask=None, encode_mask=None):
        encode_outputs, target_inputs = inputs

        output_attn_output = self.decode_multi_head([target_inputs, target_inputs, target_inputs], masks=self_mask)
        output_attn_output = self.decode_dropout(output_attn_output)
        output1 = Add()([output_attn_output, target_inputs])
        output1 = self.decode_norm(output1)
        middle_attn_output = self.encode_multi_head([output1, encode_outputs, encode_outputs], masks=encode_mask)
        middle_attn_output = self.encode_dropout(middle_attn_output)
        output2 = Add()([middle_attn_output, output1])
        output2 = self.encode_norm(output2)
        feed_output = self.point_wise(output2)
        feed_output = self.point_wise_dropout(feed_output)
        output3 = Add()([feed_output, output2])
        output3 = self.point_wise_norm(output3)
        return output3


class Encode(object):
    def __init__(self, num_layers, num_head, head_dim, model_dim, inner_dim, dropout):
        self.num_layers = num_layers
        self.num_head = num_head
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.inner_dim = inner_dim
        self.dropout = dropout
        self.encode_layers = [EncoderUnit(num_head=self.num_head, head_dim=self.head_dim,
                                          model_dim=self.model_dim, inner_dim=self.inner_dim,
                                          dropout=self.dropout) for _ in range(self.num_layers)]

    def __call__(self, inputs, masks=None):
        for layer in self.encode_layers:
            inputs = layer(inputs, masks=masks)
        return inputs


class Decode(object):
    def __init__(self, num_layers, num_head, head_dim, model_dim, inner_dim, dropout):
        self.num_layers = num_layers
        self.num_head = num_head
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.inner_dim = inner_dim
        self.dropout = dropout
        self.decode_layers = [DecoderUnit(num_head=self.num_head, head_dim=self.head_dim,
                                          model_dim=self.model_dim, inner_dim=self.inner_dim,
                                          dropout=self.dropout) for _ in range(self.num_layers)]

    def __call__(self, inputs, self_mask=None, encode_mask=None):
        outputs, encoder_outputs = inputs
        for layer in self.decode_layers:
            outputs = layer([encoder_outputs, outputs], self_mask=self_mask, encode_mask=encode_mask)
        return outputs
