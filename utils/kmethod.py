# coding: utf-8

"""
Keras methods
"""

from keras import backend as K


def broadcast_axis(x, axis=-1, d=1):
    if d > 1:
        return K.concatenate([K.expand_dims(x, axis=axis) for _ in range(d)], axis=axis)
    else:
        return K.expand_dims(x, axis=axis)
