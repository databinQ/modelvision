# coding: utf-8

from keras import backend as K
from keras.engine import Model
from keras.models import Sequential
from keras.layers import Reshape, Concatenate
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Input, Embedding, TimeDistributed

from layers.dropout import DecayingDropout


class DIINModel(Model):
    def __init__(self, p=None, h=None, use_word_embedding=True, word_embedding_weights=None,
                 train_word_embeddings=False, dropout_init_keep_rate=1.0, dropout_decay_interval=10000,
                 dropout_decay_rate=0.977, use_chars=False, chars_per_word=16, char_input_dim=100,
                 char_embedding_size=8, char_conv_filters=100, char_conv_kernel_size=5,
                 use_syntactical_features=False, syntactical_feature_size=50,
                 use_exact_match=False,
                 inputs=None, outputs=None, name="DIIN"):
        """Densely Interactive Inference Network(DIIN)

        Model from paper `Natural Language Inference over Interaction Space`
        (https://openreview.net/forum?id=r1dHXnH6-&noteId=r1dHXnH6-)

        :param p: sequence length of premise
        :param h: sequence length of hypothesis
        :param use_word_embedding: whether or not to include word vectors in the model
        :param use_chars: whether or not to include character embeddings in the model
        :param use_syntactical_features: whether or not to include syntactical features (POS tags) in the model
        :param use_exact_match: whether or not to include exact match features in the model
        :param word_embedding_weights: matrix of weights for word embeddings(pre-trained vectors)
        :param train_word_embeddings: whether or not to modify word embeddings while training
        :param dropout_init_keep_rate: initial keep rate of dropout
        :param dropout_decay_interval: the number of steps to wait for the next turn update, steps means single batch,
        other than epoch
        :param dropout_decay_rate: how much to change dropout at each interval
        :param chars_per_word: how many chars are there per one word
        :param char_input_dim: character unique numbers
        :param char_embedding_size: output size of the character-embedding layer
        :param char_conv_filters: filters of the kernel applied on character embeddings
        :param char_conv_kernel_size: size of the kernel applied on character embeddings
        :param syntactical_feature_size: size of the syntactical feature vector for each word

        :param inputs: inputs of keras models
        :param outputs: outputs of keras models
        :param name: models name
        """

        if inputs or outputs:
            super(DIINModel, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        if use_word_embedding:
            assert word_embedding_weights is not None, "Word embedding weights are needed"

        inputs = []
        premise_features = []
        hypothesis_features = []

        """Embedding layer"""
        # Input: word embedding
        if use_word_embedding:
            premise_word_input = Input(shape=(p,), dtype="int64", name="premise_word_input")
            hypothesis_word_input = Input(shape=(h,), dtype="int64", name="hypothesis_word_input")
            inputs.append(premise_word_input)
            inputs.append(hypothesis_word_input)

            word_embedding = Embedding(input_dim=word_embedding_weights.shape[0],
                                       output_dim=word_embedding_weights.shape[1],
                                       weights=word_embedding_weights,
                                       trainable=train_word_embeddings,
                                       name="word_embedding")
            premise_word_embedding = word_embedding(premise_word_input)
            hypothesis_word_embedding = word_embedding(hypothesis_word_input)

            premise_word_embedding = DecayingDropout(init_keep_rate=dropout_init_keep_rate,
                                                     decay_interval=dropout_decay_interval,
                                                     decay_rate=dropout_decay_rate,
                                                     name="premise_word_dropout")(premise_word_embedding)
            hypothesis_word_embedding = DecayingDropout(init_keep_rate=dropout_init_keep_rate,
                                                        decay_interval=dropout_decay_interval,
                                                        decay_rate=dropout_decay_rate,
                                                        name="hypothesis_word_dropout")(hypothesis_word_embedding)

            premise_features.append(premise_word_embedding)
            hypothesis_features.append(hypothesis_word_embedding)

        # Input: character embedding
        if use_chars:
            premise_char_input = Input(shape=(p, chars_per_word), dtype="int64", name="premise_char_input")
            hypothesis_char_input = Input(shape=(h, chars_per_word), dtype="int64", name="hypothesis_char_input")
            inputs.append(premise_char_input)
            inputs.append(hypothesis_char_input)

            # Share weights of character-level embedding for premise and hypothesis
            character_embedding = TimeDistributed(Sequential([
                Embedding(input_dim=char_input_dim, output_dim=char_embedding_size, input_length=chars_per_word),
                Conv1D(filters=char_conv_filters, kernel_size=char_conv_kernel_size),
                GlobalMaxPooling1D(),
            ]), name="char_embedding")
            character_embedding.build(input_shape=(None, None, chars_per_word))  # Set input shape

            premise_char_embedding = character_embedding(premise_char_input)
            hypothesis_char_embedding = character_embedding(hypothesis_char_input)
            premise_features.append(premise_char_embedding)
            hypothesis_features.append(hypothesis_char_embedding)

        # Input: syntactical features
        if use_syntactical_features:
            premise_syntactical_input = Input(shape=(p, syntactical_feature_size), name="premise_syntactical_input")
            hypothesis_syntactical_input = Input(shape=(h, syntactical_feature_size),
                                                 name="hypothesis_syntactical_input")
            inputs.append(premise_syntactical_input)
            inputs.append(hypothesis_syntactical_input)
            premise_features.append(premise_syntactical_input)
            hypothesis_features.append(hypothesis_syntactical_input)

        # Input: one-hot exact match feature
        if use_exact_match:
            premise_exact_match_input = Input(shape=(p,), name='premise_exact_match_input')
            hypothesis_exact_match_input = Input(shape=(h,), name='hypothesis_exact_match_input')
            inputs.append(premise_exact_match_input)
            inputs.append(hypothesis_exact_match_input)

            premise_exact_match = Reshape(target_shape=(p, 1))(premise_exact_match_input)
            hypothesis_exact_match = Reshape(target_shape=(h, 1))(hypothesis_exact_match_input)
            premise_features.append(premise_exact_match)
            hypothesis_features.append(hypothesis_exact_match)

        # Concatenate all features
        premise_embedding = Concatenate()(premise_features)
        hypothesis_embedding = Concatenate()(hypothesis_features)
        d = K.int_shape()[-1]

        """Encoding Layer"""




