# coding: utf-8

import argparse
import numpy as np

from constants import DATA_PATH
from train import NLITaskTrain
from models.diin import DIINModel


import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_PATH + "paipaidai/")
    parser.add_argument("--use_word_embedding", action="store_true")
    parser.add_argument("--use_chars", action="store_false")
    parser.add_argument("--use_syntactical_features", action="store_false")
    parser.add_argument("--use_exact_match", action="store_false")
    parser.add_argument("--train_word_embeddings", action="store_false")
    parser.add_argument("--dropout_init_keep_rate", type=float, default=1.0)
    parser.add_argument("--dropout_decay_interval", type=int, default=10000)
    parser.add_argument("--dropout_decay_rate", type=float, default=0.977)
    parser.add_argument("--first_scale_down_ratio", type=float, default=0.3)
    parser.add_argument("--nb_dense_blocks", type=int, default=3)
    parser.add_argument("--layers_per_dense_block", type=int, default=8)
    parser.add_argument("--nb_labels", type=int, default=2)
    parser.add_argument("--growth_rate", type=int, default=20)
    parser.add_argument("--transition_scale_down_ratio", type=float, default=0.5)

    args = parser.parse_args()

    data_dir = args.data_dir
    word_embedding_dir = data_dir + "word_embedding.npy"
    train_premise_dir = data_dir + "train_word1.npy"
    train_hypothesis_dir = data_dir + "train_word2.npy"
    test_premise_dir = data_dir + "test_word1.npy"
    test_hypothesis_dir = data_dir + "test_word2.npy"

    word_embedding = np.load(word_embedding_dir)
    train_premise = np.load(train_premise_dir)
    train_hypothesis = np.load(train_hypothesis_dir)
    test_premise = np.load(test_premise_dir)
    test_hypothesis = np.load(test_hypothesis_dir)

    model = DIINModel(p=train_premise.shape[1],
                      h=train_hypothesis.shape[1],
                      use_word_embedding=args.use_word_embedding,
                      use_chars=args.use_chars,
                      use_syntactical_features=args.use_syntactical_features,
                      use_exact_match=args.use_exact_match,
                      word_embedding_weights=word_embedding,
                      train_word_embeddings=args.train_word_embeddings,
                      dropout_init_keep_rate=args.dropout_init_keep_rate,
                      dropout_decay_interval=args.dropout_decay_interval,
                      dropout_decay_rate=args.dropout_decay_rate,
                      first_scale_down_ratio=args.first_scale_down_ratio,
                      nb_dense_blocks=args.nb_dense_blocks,
                      layers_per_dense_block=args.layers_per_dense_block,
                      growth_rate=args.growth_rate,
                      transition_scale_down_ratio=args.transition_scale_down_ratio,
                      nb_labels=args.nb_labels,)

    a = 1

