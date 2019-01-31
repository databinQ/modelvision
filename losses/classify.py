# coding: utf-8


import tensorflow as tf
from keras import backend as K


def focal_loss(gamma=2, alpha=None):
    def _focal_loss(label, logit, num_classes=2):
        """
        :param label: (batch_size,)
        :param logit: (batch_size, classes_num)
        """
        label = K.one_hot(label, num_classes=num_classes)
        one_minus_p = tf.where(tf.equal(label, tf.ones_like(label)), label - logit, tf.zeros_like(label))
        fl = -1 * one_minus_p ** gamma * tf.log(tf.clip_by_value(one_minus_p, 1e-8, 1.0))

        if alpha is None:
            return tf.reduce_sum(fl)

        assert len(alpha) == num_classes
        t_alpha = tf.convert_to_tensor(alpha, dtype=logit.dtype)
        t_alpha = tf.reshape(t_alpha, shape=(1, -1))
        alpha_ = tf.zeros_like(logit, dtype=logit.dtype) + t_alpha
        fl = alpha_ * fl
        return tf.reduce_sum(fl)

    return _focal_loss


def robust_cross_entropy(e=0.1):
    def _func(y_true, y_pred):
        classes_num = K.int_shape(y_pred)[-1]
        return (1 - e) * K.categorical_crossentropy(y_true, y_pred) + e * K.categorical_crossentropy(
            tf.ones_like(y_pred) / classes_num, y_pred)
    return _func
