'''
utils.py

Contains various utilities used all over the framework including functions for preprocessing, visualization, and so on.
'''
import code
import unicodedata
import re

import torch


class Language:
    def __init__(self, name):
        self.name = name
        # Start the vocabulary with the two special tokens
        self.word2index = {'<bos>': [0, 1], '<eos>': [1, 1]}  # value: [index, freq]
        self.index2word = ['<bos>', '<eos>']

    def add_sentence(self, sent):
        # Read a sentence, update the vocabulary
        for word in sent.split():
            if word not in self.word2index:
                self.word2index[word] = [len(self.index2word), 1]
                self.index2word.append(word)
            else:
                self.word2index[word][1] += 1

    def to_tensor(self, sent):
        t = [self.word2index[w][0] for w in sent.split()] + [1]  # 1 for eos
        return torch.tensor(t, dtype=torch.long).view(-1, 1)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


"""
Data folder: ./data or ./datasets
You can create a data folder where you can save resource files. Some people even remove 'w' permission from this folder
and only allow reading.

You can also create a data script: data_loader.py,
which includes necessary functions to find and create the right datasets as well as custom data loaders to forward the 
data to the training pipeline. You can create a base class to represent a dataset.
"""
def prepare_data(in_lang, out_lang, max_length, device):
    pairs_raw = []
    pairs_tensor = []
    in_lm = Language(in_lang)
    out_lm = Language(out_lang)
    # Read the specified dataset
    print('Reading lines...')
    with open(f'data/{in_lang}-{out_lang}.txt', encoding='utf-8') as f:
        for line in f:
            pairs_raw.append([normalizeString(l[:max_length]) for l 
                              in line.strip().split('\t')[:2]])

            # While reading construct language models for both languages
            in_lm.add_sentence(pairs_raw[-1][0])
            out_lm.add_sentence(pairs_raw[-1][1])
    for sent1, sent2 in pairs_raw:
        pairs_tensor.append((in_lm.to_tensor(sent1).to(device), 
                             out_lm.to_tensor(sent2).to(device)))

    return in_lm, out_lm, pairs_tensor
