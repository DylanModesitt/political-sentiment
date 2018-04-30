# system
import warnings
from collections import OrderedDict
from keras.preprocessing.text import text_to_word_sequence

# lib
import numpy as np


class Tokenizer:
    """Text tokenization utility class. This is adopted from keras
    and improved to behave in a more normal fashion.

    This class allows to vectorize a text corpus, by turning each
    text into either a sequence of integers (each integer being the index
    of a token in a dictionary) or into a vector where the coefficient
    for each token could be binary, based on word count, based on tf-idf...

    # Arguments
        num_words: the maximum number of words to keep, based
            on word frequency. Only the most common `num_words` words will
            be kept. If an oov_token is supplied, then any words after this
            will be tied to the oov token.
        filters: a string where each element is a character that will be
            filtered from the texts. The default is all punctuation, plus
            tabs and line breaks, minus the `'` character.
        lower: boolean. Whether to convert the texts to lowercase.
        split: character or string to use for token splitting.
        char_level: if True, every character will be treated as a token.
        oov_token: if given, it will be added to word_index and used to
            replace out-of-vocabulary words during text_to_sequence calls

    By default, all punctuation is removed, turning the texts into
    space-separated sequences of words
    (words maybe include the `'` character). These sequences are then
    split into lists of tokens. They will then be indexed or vectorized.

    `0` is a reserved index that won't be assigned to any word.
    """

    def __init__(self, num_words=None,
                 filters='"#$%&()*+-/:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True,
                 split=' ',
                 char_level=False,
                 oov_token=None):

        self.word_counts = OrderedDict()
        self.word_docs = {}
        self.filters = filters
        self.split = split
        self.lower = lower
        self.num_words = num_words
        self.document_count = 0
        self.char_level = char_level
        self.oov_token = oov_token
        self.index_docs = {}

    def fit_on_texts(self, texts):
        """Updates internal vocabulary based on a list of texts.
        In the case where texts contains lists, we assume each entry of the lists
        to be a token.

        Required before using `texts_to_sequences` or `texts_to_matrix`.

        # Arguments
            texts: can be a list of strings,
                a generator of strings (for memory-efficiency),
                or a list of list of strings.
        """
        for text in texts:
            self.document_count += 1
            if self.char_level or isinstance(text, list):
                seq = text
            else:
                seq = text_to_word_sequence(text,
                                            self.filters,
                                            self.lower,
                                            self.split)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                if w in self.word_docs:
                    self.word_docs[w] += 1
                else:
                    self.word_docs[w] = 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]

        # use only the num words desired
        sorted_voc = sorted_voc[:self.num_words-1]

        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

        if self.oov_token is not None:
            i = self.word_index.get(self.oov_token)
            if i is None:
                self.word_index[self.oov_token] = len(self.word_index) + 1
            else:
                raise ValueError('your oov token appears in the corpus'
                                 'that is probably unintended.')

        for w, c in list(self.word_docs.items()):
            val = self.word_index.get(w, self.word_index.get(self.oov_token, None))
            if val is not None:
                self.index_docs[val] = c

    def texts_to_sequences(self, texts):
        """Transforms each text in texts in a sequence of integers.

        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            texts: A list of texts (strings).

        # Returns
            A list of sequences.
        """
        return [v for v in self.texts_to_sequences_generator(texts)]

    def texts_to_sequences_generator(self, texts):
        """Transforms each text in `texts` in a sequence of integers.
        Each item in texts can also be a list, in which case we assume each item of that list
        to be a token.

        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            texts: A list of texts (strings).

        # Yields
            Yields individual sequences.
        """
        num_words = self.num_words
        for text in texts:
            if self.char_level or isinstance(text, list):
                seq = text
            else:
                seq = text_to_word_sequence(text,
                                            self.filters,
                                            self.lower,
                                            self.split)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        continue
                    else:
                        vect.append(i)
                elif self.oov_token is not None:
                    i = self.word_index.get(self.oov_token)
                    if i is not None:
                        vect.append(i)
            yield vect
