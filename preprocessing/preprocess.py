# system
import sys
import re
from dataclasses import dataclass
from typing import Sequence, Mapping

# lib
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# self
import data.ibc.treeUtil as treeUtil
sys.modules['treeUtil'] = treeUtil
from data.ibc.data import get_ibc_data


@dataclass
class Data:

    x_train: Sequence
    y_train: Sequence
    x_test: Sequence
    y_test: Sequence

    word_to_index: Mapping
    index_to_word: Mapping

    w_train: Sequence = None
    w_test: Sequence = None


def process_data(samples,
             labels,
             vocab_size=8000,
             max_len=50,
             oov_token='oov',
             shuffle=True,
             validation_split=0.2,
             one_hot_labels=False,
             verbose=1):
    """
    get properly formatted data and
    data tools for a given set of
    samples and labels.

    :param samples: a list of sentences (or
                    sentence fragements)
    :param labels: a list of the labels of those
                   sentences (or fragements)
    :param verbose: logging level of data creation

    :return: an intantiation of the above dataclass
    """
    assert len(samples) == len(labels)

    ##########################
    ## Text Cleaning
    #########################

    # standardize case
    samples = [sample.lower() for sample in samples]

    # replace space-like characters
    samples = [s.replace('\/', ' ') for s in samples]
    samples = [s.replace('-', ' ') for s in samples]

    # replace numbers with 'num' token as much political text will have numbers
    samples = [re.sub('\d+', 'num', s) for s in samples]

    # remove invalid characters
    whitelist = set('abcdefghijklmnopqrstuvwxyz!?., ')
    samples = [''.join([c for c in s if c in whitelist]) for s in samples]

    # add bos and eos tokens
    samples = ['bos ' + s + ' eos' for s in samples]

    # make all sentence-ending punctuation have a space before it to properly tokenize
    samples = [re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', s) for s in samples]

    ##########################
    # Tokenization
    #########################

    t = Tokenizer(num_words=vocab_size,
                  oov_token=oov_token,
                  filters='"#$%&()*+-/:;<=>?@[\\]^_`{|}~\t\n')

    t.fit_on_texts(samples)

    word_to_index = t.word_index
    index_to_word = {v: k for k, v in word_to_index.items()}

    x = t.texts_to_sequences(samples)
    x = pad_sequences(x, padding='pre', maxlen=max_len)

    x = np.array(x)
    y = np.array(labels)

    if shuffle:
        p = np.random.permutation(len(x))
        x = x[p]
        y = y[p]

    split_idx = int(validation_split*len(x))

    x_train = x[split_idx:]
    x_test = x[:split_idx]
    y_train = y[split_idx:]
    y_test = y[:split_idx]

    data = Data(
        x_train,
        y_train,
        x_test,
        y_test,
        word_to_index,
        index_to_word
    )

    return data


if __name__ == '__main__':
    X, Y = get_ibc_data()
    process_data(X, Y)








