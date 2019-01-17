# coding: utf-8

from keras.layers import *
from keras import backend as K
from keras.engine import Model

from layers.transformer import *


class Transformer(Model):
    def __init__(self, source_len, target_len, scr_dict_size, tar_dict_size, num_layers=6, model_dim=512, num_head=8,
                 head_dim=None, inner_dim=2048, use_pos_embedding=True, use_mask=False,
                 inputs=None, outputs=None, name="Transformer"):
        if inputs is not None and outputs is not None:
            super(Transformer, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        self.source_len = source_len
        self.target_len = target_len
        self.scr_dict_size = scr_dict_size
        self.tar_dict_size = tar_dict_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.model_dim = model_dim
        self.head_dim = head_dim if head_dim is not None else int(model_dim / num_head)
        self.inner_dim = inner_dim
        self.use_mask = use_mask
        self.use_pos_embedding = use_pos_embedding

        source_input = Input(shape=(self.source_len,), dtype="int32")
        target_input = Input(shape=(self.target_len,), dtype="int32")

        target_decode_in = Lambda(
            lambda x: K.slice(x, start=[0, 0], size=[K.shape(target_input)[0], K.shape(target_input)[1] - 1])
        )(target_input)

        if use_mask:
            source_len = Lambda(lambda x: K.sum(K.cast(K.not_equal(x, 0), dtype="int32"), axis=1))(source_input)
            target_len = Lambda(lambda x: K.sum(K.cast(K.not_equal(x, 0), dtype="int32"), axis=1))(target_decode_in)
        else:
            source_len = target_len = None

        source_embedding = Embedding(input_dim=self.scr_dict_size, output_dim=self.model_dim)
        target_embedding = Embedding(input_dim=self.tar_dict_size, output_dim=self.model_dim)
        position_embedding = PositionEmbedding()

        source_in = source_embedding(source_input)
        if self.use_pos_embedding:
            source_in = position_embedding(source_in)

        encode_output = Encode(num_layers=self.num_layers, num_head=self.num_head, head_dim=self.head_dim,
                               model_dim=self.model_dim, inner_dim=self.inner_dim)(source_in, seq_len=source_len)

        target_in = target_embedding(target_decode_in)
        if self.use_pos_embedding:
            target_in = position_embedding(target_in)

        target_output = Decode(num_layers=self.num_layers, num_head=self.num_head, head_dim=self.head_dim,
                               model_dim=self.model_dim, inner_dim=self.inner_dim)(
            [target_in, encode_output], input_seq_len=source_len, output_seq_len=target_len)

        output = TimeDistributed(Dense(self.tar_dict_size))(target_output)
        super(Transformer, self).__init__(inputs=[source_input, target_input], outputs=output, name=name)
