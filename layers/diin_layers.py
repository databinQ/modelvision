# coding: utf-8

from keras import backend as K
from keras.engine import Layer
from keras.activations import softmax

from utils.kmethod import broadcast_axis
from layers.dropout import DecayingDropout


class Encoding(Layer):
    def __init__(self, **kwargs):
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.b1 = None
        self.b2 = None
        self.b3 = None
        self.w_atten = None  # Weights of self-attention layer
        super(Encoding, self).__init__(**kwargs)

    def build(self, input_shape):
        d = input_shape[-1]

        """glorot_uniform: special uniform with limit `scale = sqrt(6 / (f_in + f_out))`, where f_in, f_out are
        matrix shape[0] and shape[1]
        """
        self.w_atten = self.add_weight(name="weights_attention", shape=(3 * d,), initializer="glorot_uniform")
        self.w1 = self.add_weight(name="w1", shape=(2 * d, d,), initializer="glorot_uniform")
        self.w2 = self.add_weight(name="w2", shape=(2 * d, d,), initializer="glorot_uniform")
        self.w3 = self.add_weight(name="w3", shape=(2 * d, d,), initializer="glorot_uniform")
        self.b1 = self.add_weight(name='b1', shape=(d,), initializer='zeros')
        self.b2 = self.add_weight(name='b2', shape=(d,), initializer='zeros')
        self.b3 = self.add_weight(name='b3', shape=(d,), initializer='zeros')

        self.w_atten.penalize_difference = "w_atten"
        self.w1.penalize_difference = "w1"
        self.w2.penalize_difference = "w2"
        self.w3.penalize_difference = "w3"
        self.b1.penalize_difference = "b1"
        self.b2.penalize_difference = "b2"
        self.b3.penalize_difference = "b3"
        super(Encoding, self).build(input_shape)

    def call(self, P, **kwargs):
        """
        Input P's shape is (batch_size, p, d), and the output shape of encoding results is (batch_size, 3 * d, p).

        For each time stamp vector pairs in P, like (P[i], P[j]), expand them in three different ways:
        1. up: first d dimensions represent P[i]
        2. mid: middle d dimensions represent P[j]
        3. down: last d dimensions represent P[i] * P[j]

        After above processing, a new tensor with shape (p, p, 3d) will be created.

        Then, use an weight vector with shape (3d,) apply on above tensor, resulting an (p, p) attention matrix, which
        means, for each token in P, the weights of every token impact on itself.

        Broadcast plays an important role on this method.
        """

        """Alpha"""
        # P                                                     # (batch, p, d)
        _, p, _ = K.int_shape(P)
        up = broadcast_axis(P, axis=-1, d=p)                    # (batch, p, d, p)
        mid = K.permute_dimensions(up, pattern=(0, 3, 2, 1))    # (batch, p, d, p)
        alphaP = K.concatenate([up, mid, up * mid], axis=2)     # (batch, p, 3d, p)
        A = K.dot(self.w_atten, alphaP)                         # (batch, p, p)

        "Self-attention"
        SA = softmax(A, axis=2)         # (batch, p, p)
        attn_res = K.batch_dot(SA, P)   # (batch, p, d)

        """Fuse gate"""
        P_concat = K.concatenate([P, attn_res], axis=2)                         # (batch, p, 2d)
        z = K.tanh(K.dot(DecayingDropout()(P_concat), self.w1) + self.b1)       # (batch, p, d)
        r = K.sigmoid(K.dot(DecayingDropout()(P_concat), self.w2) + self.b2)    # (batch, p, d)
        f = K.sigmoid(K.dot(DecayingDropout()(P_concat), self.w3) + self.b3)    # (batch, p, d)

        encoding = r * P + f * z    # (batch, p, d)
        return encoding             # (batch, p, d)


class Interaction(Layer):
    def call(self, inputs, **kwargs):
        """
        Perform element-wise multiplication for each row of premise and hypothesis.

        For every i, j, betta(premise[i], premise[j]) = premise[i] * premise[j], and get the result tensor with shape
        (batch, p, h, d).

        Use broadcast to achieve the goal.
        """

        assert len(inputs) == 2, "Number of inputs must equals to 2"
        premise, hypothesis = inputs

        premise = K.expand_dims(premise, axis=2)            # (batch, p, 1, d)
        hypothesis = K.expand_dims(hypothesis, axis=1)      # (batch, 1, h, d)
        return premise * hypothesis

    def compute_output_shape(self, input_shape):
        premise_shape = input_shape[0]
        hypothesis_shape = input_shape[1]

        assert len(premise_shape) == len(hypothesis_shape) == 3
        assert premise_shape[0] == hypothesis_shape[0]
        assert premise_shape[2] == hypothesis_shape[2]

        batch = premise_shape[0]
        p = premise_shape[1]
        h = hypothesis_shape[1]
        d = hypothesis_shape[2]
        return batch, p, h, d
