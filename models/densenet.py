# coding: utf-8

from collections import Iterable

import keras.backend as K
from keras.engine import Model
from keras.layers.merge import concatenate
from keras.layers import Input, BatchNormalization, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
# from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import imagenet_utils

_obtain_input_shape = imagenet_utils._obtain_input_shape


class DenseNet(Model):
    def __init__(self, input_shape=None, include_top=True, input_tensor=None, depth=40, nb_dense_block=3,
                 nb_layers_per_block=-1, growth_rate=12, dropout_rate=0, compression=1.0, apply_batch_norm=False,
                 classes=10, activation="softmax", inputs=None, outputs=None, name="DenseNet"):
        """https://arxiv.org/abs/1608.06993

        :param input_shape: optional shape tuple, only to be specified if `include_top` is False
        :param include_top: whether to include the fully-connected layer at the top of the network
        :param input_tensor: optional keras tensor to use as image input for the model
        :param depth: number or layers in the DenseNet
        :param growth_rate: number of filters to add per dense block
        :param nb_dense_block: number of dense blocks to add to end (generally = 3)
        :param nb_layers_per_block: number of layers in each dense block, can be a -1, positive integer or a list
            If -1, calculates nb_layer_per_block from the network depth;
            If positive integer, a set number of layers per dense block;
            If list, nb_layer is used as provided. Note that list size must be (nb_dense_block + 1).
        :param dropout_rate: dropout rate
        :param compression: scale down ratio of feature maps
        :param apply_batch_norm: whether or not use batch normalization in dense block
        :param classes: optional number of classes to classify images into, only to be specified if `include_top`
         is True
        :param activation: Type of activation at the top layer, must be one of `sigmoid` or `softmax`, note that
         if sigmoid is used, classes must be 1
        """

        if inputs or outputs:
            super(DenseNet, self).__init__(inputs, outputs, name=name)

        if activation not in ["softmax", "sigmoid"]:
            raise ValueError("activation must be one of `softmax` or `sigmoid`")

        if activation == 'sigmoid' and classes != 1:
            raise ValueError("sigmoid activation can only be used when classes == 1")

        """Determine proper input shape"""
        input_shape = _obtain_input_shape(input_shape, default_size=32, min_size=8, data_format=K.image_data_format(),
                                          require_flatten=include_top)

        if input_tensor is None:
            net_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                net_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                net_input = input_tensor

        """Construct DenseNet"""
        self.concat_axis = 1 if K.image_data_format() == "channels_first" else -1
        # Compute number of layers in each dense block
        if isinstance(nb_layers_per_block, Iterable):
            nb_layers = list(nb_layers_per_block)
            assert nb_dense_block + 1 == len(nb_layers)
            nb_layers = nb_layers[:-1]
        else:
            if nb_layers_per_block == -1:
                assert (depth - 4) % 3 == 0, "Depth must be 3N + 4 if nb_layers_per_block == -1"
                count = int((depth - 4) / 3)
                nb_layers = [count for _ in range(nb_dense_block)]
            else:
                nb_layers = [nb_layers_per_block] * nb_dense_block

        x = net_input
        for block_idx in range(nb_dense_block):
            x, nb_filter = self.__dense_block(x, nb_layers[block_idx], growth_rate, dropout_rate, apply_batch_norm)
            x = self.__transition_block(x, nb_filter, compression, apply_batch_norm)

        if apply_batch_norm:
            x = BatchNormalization(axis=self.concat_axis, epsilon=1.1e-5)(x)

        try:
            x = Flatten()(x)
        except:
            x = GlobalAveragePooling2D()(x)

        if include_top:
            x = Dense(classes, activation=activation)(x)

        super(DenseNet, self).__init__(inputs=net_input, outputs=x, name=name)

    def __dense_block(self, x, nb_layers, growth_rate, dropout_rate=None, apply_batch_norm=None):
        for i in range(nb_layers):
            cb = self.__conv_block(x, growth_rate, dropout_rate, apply_batch_norm)
            x = concatenate([x, cb], axis=self.concat_axis)
        return x, K.int_shape(x)[self.concat_axis]

    def __conv_block(self, x, nb_filter, dropout_rate=None, apply_batch_norm=False):
        """Apply BatchNorm, Relu, 3x3 Conv2D, optional dropout"""
        if apply_batch_norm:
            x = BatchNormalization(axis=self.concat_axis, epsilon=1.1e-5)(x)

        x = Conv2D(filters=nb_filter, kernel_size=(3, 3), padding="same", activation="relu")(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        return x

    def __transition_block(self, x, nb_filter, compression, apply_batch_norm):
        """Apply bottleneck and Maxpool"""
        if apply_batch_norm:
            x = BatchNormalization(axis=self.concat_axis, epsilon=1.1e-5)(x)

        x = Conv2D(int(nb_filter * compression), (1, 1), padding='same', activation=None)(x)
        x = MaxPooling2D(strides=(2, 2))(x)
        return x
