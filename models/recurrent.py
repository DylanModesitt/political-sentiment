# system 
from typing import Sequence, Tuple
from dataclasses import dataclass
import os

# lib
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import (
    Input,
    Embedding,
    LSTM,
    GRU,
    RNN,
    Dense,
    Bidirectional,
    Dropout
)
from keras.regularizers import l1_l2
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback, ModelCheckpoint

# self
from models.model import PoliticalSentimentModel
from data.ibc.data import get_ibc_data
from data.twitter.data import get_congressional_twitter_data
from preprocessing.preprocess import Data, process_data
from general.embeddings import get_pretrained_embedding_matrix, GloVeSize


@dataclass
class RecurrentSentimentModel(PoliticalSentimentModel):
    """
    initialize the model by passing in model
    parameters described by the following declarations. 
    These parameters must be passed in the order provided.
    
    If the parameter does not have a default value,
    provide one to the initializer.
    """

    input_length: int = 50
    vocabulary_size: int = 10000
    recurrent_cell: RNN = LSTM

    embedding_size: int = 100
    embedding_matrix: Sequence = None

    latent_dim: int = 256
    use_bidirection: bool = True 
    dropout: int = 0.2
    regularization: Tuple[int, int] = (0.0, 0.0001)

    learning_rate: float = 0.001

    def initialize_model(self):
        """
        initialize the model give the 
        model's parameters.
        """

        input_layer = Input(
            shape=(self.input_length,), 
            dtype='int32', 
            name='input'
        )

        if self.embedding_matrix is None:
            embedding = Embedding(
                 output_dim=self.embedding_size,
                 input_dim=self.vocabulary_size + 1, # for mask
                 input_length=self.input_length,
                 mask_zero=True,
                 name='embedding'
            )(input_layer)
        else:
            embedding = Embedding(
                output_dim=self.embedding_size,
                input_dim=self.vocabulary_size + 1,
                input_length=self.input_length,
                mask_zero=True,
                weights=[np.vstack((np.zeros((1, self.embedding_size)),
                                    self.embedding_matrix))],
                name='embedding'
            )(input_layer)

        encoder = self.recurrent_cell(
            self.latent_dim,
            dropout=self.dropout,
            recurrent_dropout=self.dropout,
            name='encoder',
            recurrent_regularizer=l1_l2(*self.regularization)
        )

        if self.use_bidirection:
            encoder = Bidirectional(
                encoder,
                merge_mode='concat'
            )

        encoder = encoder(embedding)

        dense_1 = Dense(
            1024,
            activation='tanh',
            name='dense_1',
            kernel_regularizer=l1_l2(*self.regularization)
        )(encoder)

        dense_2 = Dense(
            512,
            activation='tanh',
            name='dense_2',
            kernel_regularizer=l1_l2(*self.regularization)
        )(dense_1)

        dropout = Dropout(self.dropout)(
            dense_2
        )

        prediction = Dense(
            1,
            activation='sigmoid',
            name='prediction'
        )(dropout)

        model = Model(inputs=input_layer, outputs=prediction)

        # sparse_categorical_crossentropy
        model.compile(optimizer=Adam(lr=self.learning_rate),
                      loss='binary_crossentropy',
                      metrics=['acc'])

        self.model = model

        if self.verbose > 0:
            model.summary()

        return [model]

    def __call__(self, data, epochs=100, batch_size=128):
        """
        fit the model around the given data

        :param epochs: the number of epochs to fit for
        :param batch_size: the batch size to fit with
        :return: the history
        """
        self.word_to_index = data.word_to_index
        self.save()

        def save_(a,b):
            self.save()

        global_save = LambdaCallback(on_epoch_end=save_)
        best_save = ModelCheckpoint(filepath=os.path.join(self.dir, 'weights/agent_0_best.h5'),
                                    save_weights_only=True, save_best_only=True)

        history = self.model.fit(
            data.x_train,
            data.y_train,
            validation_data=(data.x_test, data.y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[global_save, best_save]
        ).history

        self.save()

        self.history = self.generate_cohesive_history({
            'recurrent': history
        })

        return history 


def fit_ibc():

    X, Y = get_ibc_data(
        use_neutral=False,
        use_subsampling=True
    )

    data = process_data(
        X,
        Y,
        validation_split=0.1,
        max_len=RecurrentSentimentModel.input_length
    )

    embedding_size = GloVeSize.small
    embedding_matrix = get_pretrained_embedding_matrix(data.word_to_index,
                                                       size=embedding_size)

    model = RecurrentSentimentModel(embedding_size=embedding_size.value,
                                    embedding_matrix=embedding_matrix)

    model(data, epochs=10)
    model.visualize(metrics_to_mix=[('acc', 'val_acc'),
                                    ('loss', 'val_loss')])


def fit_twitter():

    X, Y = get_congressional_twitter_data(use_senate=True, use_house=True)

    data = process_data(
        X,
        Y,
        twitter=True,
        validation_split=0.1,
        max_len=RecurrentSentimentModel.input_length
    )

    embedding_size = GloVeSize.small
    embedding_matrix = get_pretrained_embedding_matrix(data.word_to_index,
                                                       size=embedding_size)

    model = RecurrentSentimentModel(embedding_size=embedding_size.value,
                                    embedding_matrix=embedding_matrix)

    model(data, epochs=10)
    model.visualize(metrics_to_mix=[('acc', 'val_acc'),
                                    ('loss', 'val_loss')])


if __name__ == '__main__':
    fit_ibc()


