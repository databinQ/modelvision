# coding: utf-8

from keras.engine import Layer


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








