# coding: utf-8

import re
import copy
import codecs
from tqdm import tqdm
from collections import defaultdict


class Subwords(object):
    def __init__(self, num_symbols=50000, min_frequency=2):
        self.num_symbols = num_symbols
        self.min_frequency = min_frequency

        self.vocabulary = defaultdict(int)
        self.vocabulary_chars = defaultdict(int)
        self.word2index = defaultdict(int)
        self.index2word = defaultdict(tuple)
        self.index2freq = defaultdict(int)
        self.pair_stats = defaultdict(int)
        self.pair_stats_total = defaultdict(int)
        self.pair2word = defaultdict(lambda: defaultdict(int))
        self.bpe = defaultdict(int)
        self.cache = dict()

    @staticmethod
    def get_tokens(texts, sep=" "):
        texts = [tokens.strip(" \r\n").split(sep) for tokens in texts]
        return [[token for token in tokens if token] for tokens in texts]

    def get_vocabulary(self, texts):
        for tokens in texts:
            for token in tokens:
                token = token.strip()
                if token:
                    self.vocabulary[token] += 1

    def _get_vocabulary_chars(self):
        for word, freq in self.vocabulary.items():
            self.vocabulary_chars[tuple(word[:-1]) + tuple([word[-1] + "</w>", ])] = freq

    def _vocab_sort(self):
        t = sorted([(k, v) for k, v in self.vocabulary_chars.items()], key=lambda x: x[1], reverse=True)
        for i, (chars, freq) in enumerate(t):
            self.word2index[chars] = i
            self.index2word[i] = chars
            self.index2freq[i] = freq

    def get_pair_stats(self):
        for i, chars in self.index2word.items():
            prev_char = chars[0]
            freq = self.index2freq[i]
            for char in chars[1:]:
                self.pair_stats[prev_char, char] += freq
                self.pair2word[prev_char, char][i] += 1
                prev_char = char

    def pair_replace(self, pair):
        first, second = pair
        pair_str = "".join(pair).replace("\\", "\\\\")
        pattern = re.compile(r"(?<!\S)" + re.escape(first + " " + second) + r"(?!\S)")
        changes = []
        for index, times in self.pair2word[pair].items():
            if times < 1:
                continue
            word_chars, freq = self.index2word[index], self.index2freq[index]
            new_word = " ".join(word_chars)
            new_word = pattern.sub(pair_str, new_word)
            new_word = tuple(new_word.split(" "))

            self.index2word[index] = new_word
            self.word2index[new_word] = index
            del self.word2index[word_chars]
            changes.append((index, new_word, word_chars, freq))
        return changes

    def update_pair_stats(self, pair, changes):
        first, second = pair
        new_subword = "".join(pair).replace("\\", "\\\\")

        self.pair_stats[pair] = 0
        self.pair2word[pair] = defaultdict(int)

        for index, new_word, word_chars, freq in changes:
            i = 0
            while 1:
                try:
                    i = word_chars.index(first, i)
                except ValueError:
                    break  # can not find more pairs to be replaced

                if i < len(word_chars) - 1 and word_chars[i + 1] == second:
                    if i > 0:
                        """
                        assuming a symbol sequence `A B C`, if `B C` is merged, reduce the frequency of `A B`
                        """
                        prev_pair = word_chars[i - 1: i + 1]
                        self.pair_stats[prev_pair] -= freq
                        self.pair2word[prev_pair][index] -= 1
                    if i < len(word_chars) - 2 and (word_chars[i + 2] != first or i >= len(word_chars) - 3 or
                                                    word_chars[i + 3] != second):
                        """
                        assuming a symbol sequence `A B C B`, if `B C` is merged, reduce the frequency of `C B`.
                        however, skip this if the sequence is `A B C B C`, because the frequency of `C B` will be
                        reduced by the previous code block.
                        """
                        next_pair = word_chars[i + 1: i + 3]
                        self.pair_stats[next_pair] -= freq
                        self.pair2word[next_pair][index] -= 1
                    i += 2
                else:
                    i += 1

            i = 0
            while 1:
                try:
                    i = new_word.index(new_subword, i)
                except ValueError:
                    break

                if i > 0:
                    """
                    assuming a symbol sequence `A BC D`, if `B C` is merged, increase the frequency of `A BC`
                    """
                    prev_pair = new_word[i - 1: i + 1]
                    self.pair_stats[prev_pair] += freq
                    self.pair2word[prev_pair][index] += 1
                if i < len(new_word) - 1 and new_word[i + 1] != new_subword:
                    """
                    assuming a symbol sequence `A BC B`, if `B C` is merged, increase the frequency of `BC B`.
                    however, if the sequence is `A BC BC`, skip this step because the count of `BC BC` will be 
                    incremented by the previous code block.
                    """
                    next_pair = new_word[i: i + 2]
                    self.pair_stats[next_pair] += freq
                    self.pair2word[next_pair][index] += 1
                i += 1

    def prune_stats(self, threshold):
        for pair, freq in list(self.pair_stats.items()):
            if freq < threshold:
                del self.pair_stats[pair]
                if freq < 0:
                    self.pair_stats_total[pair] += freq
                else:
                    self.pair_stats_total[pair] = freq

    def read_texts(self, file_path):
        with codecs.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            texts = f.readlines()
        return self.get_tokens(texts)

    def train(self, texts=None, file_path=None):
        if texts is None and file_path is not None:
            texts = self.read_texts(file_path)
        elif isinstance(texts[0], str):
            texts = self.get_tokens(texts)

        self.get_vocabulary(texts)
        self._get_vocabulary_chars()
        self._vocab_sort()
        self.get_pair_stats()
        self.pair_stats_total = copy.deepcopy(self.pair_stats)

        # threshold is inspired by Zipfian assumption, only affect speed
        threshold = max(self.pair_stats.values()) / 10
        for i in range(self.num_symbols):
            if self.pair_stats:
                most_frequent = max(self.pair_stats, key=lambda x: (self.pair_stats[x], x))

            if not self.pair_stats or (i > 0 and self.pair_stats[most_frequent] < threshold):
                """we probably missed the best pair because of pruning; go back to full statistics"""
                self.prune_stats(threshold)
                self.pair_stats = copy.deepcopy(self.pair_stats_total)
                most_frequent = max(self.pair_stats, key=lambda x: (self.pair_stats[x], x))
                # threshold is inspired by Zipfian assumption, but should only affect speed
                threshold = self.pair_stats[most_frequent] * i / (i + 10000.0)
                self.prune_stats(threshold)

            if self.pair_stats[most_frequent] < self.min_frequency:
                print("No pairs have frequency >= {}, bpe process end".format(self.min_frequency))
                break

            print(*most_frequent, len(self.bpe))
            self.bpe[most_frequent] = i
            changes = self.pair_replace(most_frequent)
            self.update_pair_stats(most_frequent, changes)
            self.pair_stats[most_frequent] = 0
            self.prune_stats(threshold)

    def process(self, seqs):
        res = []
        for text in tqdm(seqs):
            res.append(self.process_single(text))
        return res

    def process_single(self, tokens):
        res = tuple()
        for word in tokens:
            t_subwords = self.encode(word)
            res += t_subwords
        return res

    def encode(self, word):
        if word in self.cache:
            return self.cache[word]

        chars = tuple(word[:-1]) + (word[-1] + "</w>",)
        pairs = self.get_pairs(chars)
        if len(pairs) == 0:
            return tuple((word,))

        while 1:
            """combine subword pairs in ascending order of bpe index"""
            bigram = min(pairs, key=lambda x: self.bpe.get(x, float("inf")))
            if bigram not in self.bpe:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(chars):
                try:
                    j = chars.index(first, i)
                    new_word.extend(chars[i: j])
                    i = j
                except ValueError:
                    new_word.extend(chars[i:])
                    break

                if chars[i] == first and i < len(chars) - 1 and chars[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(chars[i])
                    i += 1
            chars = tuple(new_word)
            if len(chars) == 1:
                break
            else:
                pairs = self.get_pairs(chars)

        if chars[-1] == "</w>":
            chars = chars[:-1]
        if chars[-1].endswith("</w>"):
            chars = chars[:-1] + (chars[-1].replace("</w>", ""),)

        self.cache[word] = chars
        return chars

    @staticmethod
    def get_pairs(chars):
        pairs = set()
        prev_char = chars[0]
        for c in chars[1:]:
            pairs.add((prev_char, c))
            prev_char = c
        return pairs


if __name__ == "__main__":
    from constants import DATA_PATH

    model = Subwords()
    texts = model.read_texts(DATA_PATH + "subword_corpus.en")
    model.train(texts=texts)
    subwords = model.process(texts)
    print(subwords[0])
