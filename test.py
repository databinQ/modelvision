# coding: utf-8

from keras.layers import Input, Conv1D
from keras.models import Model
from layers.transformer import MultiHeadAttention, EncoderUnit


inputs = Input(shape=(None, 512), dtype="float32")

# output = MultiHeadAttention(num_head=8, head_dim=64, model_dim=512, dropout=0.1)([inputs, inputs, inputs])
output = EncoderUnit(num_head=8, head_dim=64, model_dim=512, dropout=0.1, inner_dim=512)(inputs)

model = Model(inputs, output, name="test")
model.compile("adam", loss="mse")
model.summary()
a = 1
