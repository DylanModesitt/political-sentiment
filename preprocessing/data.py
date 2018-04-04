# system 
from dataclasses import dataclass, field 
from typing import Sequence, Mapping

# lib 
import numpy as np 
from keras.preprocessing.text import Tokenizer


@dataclasse
class Data:

    x_train: Sequence
    y_train: Sequence
    x_test: Sequence
    y_test: Sequence
    
    word_to_index: Mapping
    index_to_word: Mapping    

def get_data(samples,
             labels, 
             vocab_size=8000,
             oov_token='oov',
             shuffle=False,
             validation_split=0.2,
             verbose=1)
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
    assert len(samples) == len(lables)

    t = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    t.fit_on_texts(samples)

    word_to_index = t.word_index
    index_to_word = {v:k for k,v in word_to_index.items()}

    x = t.texts_to_sequences(samples)
    x = np.array(x)
    y = np.array(labels)

    if shuffle:
        p = np.random.permutation(len(x))
        x = x[p]
        y = y[p]

    split_idx = int(validation_split*len(x))

    x_train = x[split_idx:]
    x_test = x[split_idx:]
    y_train = y[split_idx:]
    y_test = y[split_idx:]

    data = Data(x_train, y_train, x_test, y_test, word_to_index, index_to_word)

    return data 


    

    
    






