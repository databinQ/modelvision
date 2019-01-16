# coding: utf-8

import argparse
import numpy as np

from constants import DATA_PATH
from train import NLITaskTrain
from models.diin import DIINModel
from optimizers.l2_optimizer import L2Optimizer
from keras.optimizers import Adam, Adagrad, SGD
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """Model parameters"""
    parser.add_argument("--data_dir", type=str, default=DATA_PATH + "paipaidai/")
    parser.add_argument("--log_dir", type=str, default=DATA_PATH + "paipaidai/logs/")
    parser.add_argument("--save_dir", type=str, default=DATA_PATH + "paipaidai/models/")
    parser.add_argument("--use_word_embedding", action="store_true", default=True)
    parser.add_argument("--use_chars", action="store_false", default=False)
    parser.add_argument("--use_syntactical_features", action="store_false", default=False)
    parser.add_argument("--use_exact_match", action="store_false", default=False)
    parser.add_argument("--train_word_embeddings", action="store_false", default=False)
    parser.add_argument("--dropout_init_keep_rate", type=float, default=1.0)
    parser.add_argument("--dropout_decay_interval", type=int, default=10000)
    parser.add_argument("--dropout_decay_rate", type=float, default=0.977)
    parser.add_argument("--first_scale_down_ratio", type=float, default=0.3)
    parser.add_argument("--nb_dense_blocks", type=int, default=3)
    parser.add_argument("--layers_per_dense_block", type=int, default=8)
    parser.add_argument("--nb_labels", type=int, default=2)
    parser.add_argument("--growth_rate", type=int, default=20)
    parser.add_argument("--transition_scale_down_ratio", type=float, default=0.5)
    """Optimizer parameters"""
    parser.add_argument("--l2_steps", type=int, default=100000)
    parser.add_argument("--l2_ratio", type=float, default=9e-5)
    parser.add_argument("--l2_difference_ratio", type=float, default=1e-3)
    """Train parameters"""
    parser.add_argument("--dev_rate", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_interval", type=int, default=1024)
    parser.add_argument("--shuffle", action="store_true", default=True)

    args = parser.parse_args()

    data_dir = args.data_dir
    word_embedding_dir = data_dir + "word_embedding.npy"
    train_premise_dir = data_dir + "train_word1.npy"
    train_hypothesis_dir = data_dir + "train_word2.npy"
    train_label_dir = data_dir + "train_label.npy"
    test_premise_dir = data_dir + "test_word1.npy"
    test_hypothesis_dir = data_dir + "test_word2.npy"

    word_embedding = np.load(word_embedding_dir)
    train_premise = np.load(train_premise_dir)
    train_hypothesis = np.load(train_hypothesis_dir)
    train_label = np.load(train_label_dir)
    test_premise = np.load(test_premise_dir)
    test_hypothesis = np.load(test_hypothesis_dir)

    # Optimizer
    adam = L2Optimizer(Adam(), args.l2_steps, args.l2_ratio, args.l2_difference_ratio)
    adagrad = L2Optimizer(Adagrad(), args.l2_steps, args.l2_ratio, args.l2_difference_ratio)
    sgd = L2Optimizer(SGD(lr=3e-3), args.l2_steps, args.l2_ratio, args.l2_difference_ratio)

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

    train_premise, dev_premise, train_hypothesis, dev_hypothesis, train_label, dev_label = train_test_split(
        train_premise, train_hypothesis, train_label, test_size=args.dev_rate)

    task = NLITaskTrain(model=model,
                        train_data=[train_premise, train_hypothesis, train_label],
                        test_data=[test_premise, test_hypothesis],
                        dev_data=[dev_premise, dev_hypothesis, dev_label],
                        optimizer=[(adam, 3), (adagrad, 4), (sgd, 15)],
                        log_dir=args.log_dir,
                        save_dir=args.save_dir)

    task.train_multi_optimizer(batch_size=args.batch_size,
                               eval_interval=len(train_label) // args.batch_size,
                               shuffle=args.shuffle)
