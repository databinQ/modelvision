# coding: utf-8

from keras import backend as K
from keras.engine import Model
from keras.models import Sequential
from keras.layers import Reshape, Concatenate
from keras.layers import Conv1D, Conv2D, GlobalMaxPooling1D
from keras.layers import Input, Embedding, TimeDistributed, Dense

from models.densenet import DenseNet
from layers.dropout import DecayingDropout
from layers.diin_layers import Encoding, Interaction


class DIINModel(Model):
    def __init__(self, p=None, h=None, use_word_embedding=True, word_embedding_weights=None,
                 train_word_embeddings=False, dropout_init_keep_rate=1.0, dropout_decay_interval=10000,
                 dropout_decay_rate=0.977, use_chars=False, chars_per_word=16, char_input_dim=100,
                 char_embedding_size=8, char_conv_filters=100, char_conv_kernel_size=5,
                 use_syntactical_features=False, syntactical_feature_size=50, use_exact_match=False,
                 first_scale_down_ratio=0.3, nb_dense_blocks=3, layers_per_dense_block=8, nb_labels=3,
                 growth_rate=20, transition_scale_down_ratio=0.5, inputs=None, outputs=None, name="DIIN"):
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
        :param first_scale_down_ratio: scale ratio of map features as the input of first Densenet block
        :param nb_dense_blocks: number of dense blocks in densenet
        :param layers_per_dense_block: number of layers in one dense block
        :param nb_labels: number of labels
        :param growth_rate:growing rate in dense net
        :param transition_scale_down_ratio: transition scale down ratio in dense net
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
                                       weights=[word_embedding_weights],
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
        if len(premise_features) > 1:
            premise_embedding = Concatenate()(premise_features)
            hypothesis_embedding = Concatenate()(hypothesis_features)
        else:
            premise_embedding = premise_features[0]
            hypothesis_embedding = hypothesis_features[0]
        d = K.int_shape(premise_embedding)[-1]

        """Encoding layer"""
        premise_encoding = Encoding(name="premise_encoding")(premise_embedding)
        hypothesis_encoding = Encoding(name="hypothesis_encoding")(hypothesis_embedding)

        """Interaction layer"""
        interaction = Interaction(name="interaction")([premise_encoding, hypothesis_encoding])

        """Feature extraction layer"""
        feature_extractor_input = Conv2D(filters=int(d * first_scale_down_ratio),
                                         kernel_size=1,
                                         activation=None,
                                         name="bottleneck")(interaction)  # Bottleneck layer
        feature_extractor = DenseNet(input_tensor=Input(shape=K.int_shape(feature_extractor_input)[1:]),
                                     include_top=False,
                                     nb_dense_block=nb_dense_blocks,
                                     nb_layers_per_block=layers_per_dense_block,
                                     growth_rate=growth_rate,
                                     compression=transition_scale_down_ratio)(feature_extractor_input)

        """Output layer"""
        features = DecayingDropout(init_keep_rate=dropout_init_keep_rate,
                                   decay_interval=dropout_decay_interval,
                                   decay_rate=dropout_decay_rate,
                                   name="features")(feature_extractor)
        if nb_labels == 2:
            out = Dense(1, activation="sigmoid", name="output")(features)
        else:
            out = Dense(nb_labels, activation="softmax", name="output")(features)
        super(DIINModel, self).__init__(inputs=inputs, outputs=out, name=name)
