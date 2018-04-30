# system
import os
from enum import Enum

# lib
import numpy as np


class GloVeSize(Enum):

    tiny = 50
    small = 100
    medium = 200
    large = 300


__DEFAULT_SIZE = GloVeSize.small


def get_pretrained_embedding_matrix(word_to_index,
                                    vocab_size=10000,
                                    glove_dir="./bin/GloVe",
                                    use_cache_if_present=True,
                                    cache_if_computed=True,
                                    cache_dir='./bin/cache',
                                    size=__DEFAULT_SIZE,
                                    verbose=1):

    """
    get pre-trained word embeddings from GloVe: https://github.com/stanfordnlp/GloVe

    :param word_to_index: a word to index map of the corpus
    :param vocab_size: the vocab size
    :param glove_dir: the dir of glove
    :param use_cache_if_present: whether to use a cached weight file if present
    :param cache_if_computed: whether to cache the result if re-computed
    :param cache_dir: the directory of the project's cache
    :param size: an enumerated choice of GloVeSize
    :param verbose: the verbosity level of logging
    :return: a matrix of the embeddings
    """
    def vprint(*args, with_arrow=True):
        if verbose > 0:
            if with_arrow:
                print(">>", *args)
            else:
                print(*args)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_path = os.path.join(cache_dir, 'glove_%d_embedding_matrix.npy' % size.value)
    if use_cache_if_present and os.path.isfile(cache_path):
        return np.load(cache_path)
    else:
        vprint('computing embeddings', with_arrow=True)
        embeddings_index = {}
        size_value = size.value
        f = open(os.path.join(glove_dir, 'glove.6B.' + str(size_value) + 'd.txt'),
                 encoding="ascii", errors='ignore')

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

        f.close()
        vprint('Found', len(embeddings_index), 'word vectors.')

        embedding_matrix = np.random.normal(size=(vocab_size, size.value))

        non = 0
        for word, index in word_to_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
            else:
                non += 1

        vprint(non, "words did not have mappings")
        vprint(with_arrow=False)

        if cache_if_computed:
            np.save(cache_path, embedding_matrix)

        return embedding_matrix


