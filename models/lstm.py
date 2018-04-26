# system 
import sys 
import os 
from dataclasses import dataclass, field

# lib 
import keras.backend as K
from keras.models import Model
from keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    Bidirectional
)

# self
from models.model import PoliticalSentimentModel
from data.ibc.data import get_ibc_data
from preprocessing.preprocess import Data, process_data

@dataclass
class LSTMSentimentModel(PoliticalSentimentModel):

    """
    initialize the model by passing in model
    parameters described by the following declarations. 
    These parameters must be passed in the order provided.
    
    If the parameter does not have a default value,
    provide one to the initializer.
    """

    input_length: int = 81
    vocabulary_size: int = 8000
    embedding_size: int = 10

    latent_dim: int = 128
    use_bidirection: bool = True 
    dropout: int = 0.3

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
        
        embedding = Embedding(
             output_dim=self.embedding_size, 
             input_dim=self.vocabulary_size,
             input_length=self.input_length,
             mask_zero=True,
             name='embedding'
        )(input_layer)

        encoder = LSTM(
            self.latent_dim,
            dropout=self.dropout,
            recurrent_dropout=self.dropout,
            name='encoder'
        )

        if self.use_bidirection:
            encoder = Bidirectional(
                encoder,
                merge_mode='concat'
            )

        encoder = encoder(embedding)

        dense = Dense(
            4*self.latent_dim,
            activation='tanh',
            name='transform'
        )(encoder)

        prediction = Dense(
            3,
            activation='softmax',
            name='prediction'
        )(dense)

        model = Model(inputs=input_layer, outputs=prediction)
        self.model = model

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

        if self.verbose > 0:
            model.summary()

        return model 

    def __call__(self, data, epochs=100, batch_size=128):
        """
        fit the model around the given data

        :param epochs: the number of epochs to fit for
        :param batch_size: the batch size to fit with
        :return: the history
        """

        history = self.model.fit(
            data.x_train,
            data.y_train,
            validation_data=(data.x_test, data.y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        ).history

        return history 


if __name__ == '__main__':

    model = LSTMSentimentModel()
    X, Y = get_ibc_data(
        use_neutral=True,
    )

    data = process_data(
        X,
        Y,
        validation_split=0.1
    )

    model(data)
