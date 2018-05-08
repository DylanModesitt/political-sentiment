# system
import sys
import re
from dataclasses import dataclass
from typing import Sequence, Mapping
from enum import Enum

# lib
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# self
from general.tokenizer import Tokenizer
import data.ibc.treeUtil as treeUtil
sys.modules['treeUtil'] = treeUtil


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


class DefaultTokens:
    """
    default replacement tokens
    for text cleaning
    """
    eos = 'eos'
    oov = 'oov'
    hashtag = '<hashtag>'
    handle = '<person>'
    

class HashtagCleanMode(Enum):
    """
    Cleaning mode for hashtags. None
    keeps them as is. Remove removes them,

    """

    NONE = 0
    REMOVE = 1
    GLOBAL = 2
    BEST_EFFORT_REPLACE = 3


class HandleCleanMode(Enum):
    """
    enumeration of modes of replacement
    for handles. That is @Username
    """

    NONE = 0
    REMOVE = 1
    GLOBAL = 2


def _add_space(match):
    pattern = "(^[a-z]+)|([A-Z]+[a-z]*)|([0-9]+)"
    phrase = match.group(1)

    if "-" in phrase:
        result = " ".join(phrase.split("-"))
        return result

    matches = re.finditer(pattern, phrase)
    result = " ".join([m.group(0) for m in matches])
    return result


def clean_tweet(sample,
                remove_hyperlinks=True,
                hashtag_mode: HashtagCleanMode = HashtagCleanMode.BEST_EFFORT_REPLACE,
                handle_mode: HandleCleanMode = HandleCleanMode.GLOBAL):
    """
    clean a tweet
    :param sample: the tweet
    :param remove_hyperlinks: remove hyperlinks?
    :param hashtag_mode: the hastag mode
    :param handle_mode: the handle mode
    :return: the cleaned tweet
    """

    # Replace '&amp;' with ' and '
    amp_pattern = re.compile("&amp;")
    sample = amp_pattern.sub(" and ", sample)

    # Replace special numbers
    num_pattern = re.compile("\d+(st|nd|rd|th)")
    sample = num_pattern.sub(" num ", sample)

    # handles
    handle_pattern = re.compile("@(\S*)")
    if handle_mode == HandleCleanMode.REMOVE:
        sample = handle_pattern.sub(" ", sample)
    elif handle_mode == HandleCleanMode.GLOBAL:
        sample = handle_pattern.sub(" %s " % DefaultTokens.handle, sample)

    # Replace hashtags
    hashtag_pattern = re.compile('#(\S*)')
    if hashtag_mode == HashtagCleanMode.REMOVE:
        sample = hashtag_pattern.sub(" ", sample)
    elif hashtag_mode == HashtagCleanMode.GLOBAL:
        sample = hashtag_pattern.sub(" %s " % DefaultTokens.hashtag, sample)
    elif hashtag_mode == HashtagCleanMode.BEST_EFFORT_REPLACE:
        sample = hashtag_pattern.sub(_add_space, sample)

    # Remove hyperlinks
    hyperlink_pattern = re.compile("https?://\S*")
    if remove_hyperlinks:
        sample = hyperlink_pattern.sub(" ", sample)

    # Remove extra spaces
    sample = " ".join(sample.split())

    return sample


def clean_text_documents(samples,
                         twitter=False,
                         remove_hyperlinks=True,
                         hashtag_mode: HashtagCleanMode = HashtagCleanMode.BEST_EFFORT_REPLACE,
                         handle_mode: HandleCleanMode = HandleCleanMode.GLOBAL):
    """
    clean a list of text documents resonably well

    :param samples: a list of text documents
    :param twitter: whether or not this is twitter data.
                    if this is false, then twitter specific
                    cleaning parameters will be ignored.
    :param remove_handles: whether to remove handles
    :param remove_hyperlinks: whether to remove hyperlinks
    :param hashtag_mode: the hashtag mode to use
    :return:
    """

    # Twitter-specific text cleaning
    if twitter:
        samples = [clean_tweet(sample, remove_hyperlinks, hashtag_mode, handle_mode) for sample in samples]

    # remove urls
    samples = [re.sub(r'https?:\/\/.*[\r\n]*', '', s) for s in samples]

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

    return samples

def process_data(samples,
                 labels,
                 twitter=False,
                 remove_hyperlinks=True,
                 hashtag_mode: HashtagCleanMode = HashtagCleanMode.BEST_EFFORT_REPLACE,
                 handle_mode: HandleCleanMode = HandleCleanMode.GLOBAL,
                 vocab_size=10000,
                 max_len=50,
                 oov_token='oov',
                 shuffle=True,
                 validation_split=0.2,
                 verbose=1):
    """
      get properly formatted data and
    data tools for a given set of
    samples and labels.

    :param samples: a list of sentences (or
                    sentence fragements)
    :param labels: a list of the labels of those
                   sentences (or fragements)
    :param verbose: logging level of data creatio
    :param twitter: if this is twitter data
    :param remove_handles: whether or not to remove handles
    :param remove_hyperlinks: whether or not to remove hyperlinks
    :param hashtag_mode: the mode to deal with hashtags
    :param vocab_size: the size of the vocab
    :param max_len: the max length
    :param oov_token: the oov token to use
    :param shuffle: whether or not to shuffle the data
    :param validation_split: the % to use for validation
    :param one_hot_labels:
    :return:
    """
    assert len(samples) == len(labels)

    def vprint(*args):
        if verbose > 0:
            print(*args)


    ##########################
    ## Text Cleaning
    #########################

    vprint('>> processing text')
    vprint('cleaning text')
    samples = clean_text_documents(samples, twitter=twitter,
                                   remove_hyperlinks=remove_hyperlinks,
                                   handle_mode=handle_mode,
                                   hashtag_mode=hashtag_mode)

    ##########################
    # Tokenization
    #########################

    t = Tokenizer(num_words=vocab_size,
                  oov_token=oov_token)

    t.fit_on_texts(samples)

    word_to_index = t.word_index
    index_to_word = {v: k for k, v in word_to_index.items()}

    x = t.texts_to_sequences(samples)
    x = pad_sequences(x, padding='pre', maxlen=max_len)

    percent_unkown = "%0.2f" % (100*np.sum(x == word_to_index[oov_token]) / np.sum(x > 0),)
    vprint(percent_unkown + '% of the words are out of vocabulary and replaced with "' + oov_token + '"')

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

