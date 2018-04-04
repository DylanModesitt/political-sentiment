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
    Dense
)


@dataclass
class Model:

    """
    initialize the model by passing in model
    parameters described by the following declarations. 
    These parameters must be passed in the order provided.
    
    If the parameter does not have a default value,
    provide one to the initializer.
    """

    input_length: int = 50
    vocabulary_size: int = 8000
    embedding_size: int = 100

    latent_dim: int = 256
    use_bidirection: bool = True 
    dropout: int = 0.2

    verbose: int = 1


    def get_model(self):
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
             name='embedding'
        )(input_layer)

        encoder = LSTM(
            self.latent_dim,
            dropout=self.dropout,
            recurrent_dropout=self.dropout,
            name='encoder'
        )(embedding)

        dense = Dense(
            4*self.latent_dim,
            activation='tanh',
            name='transform'
        )(encoder)


        prediction = Dense(
            1,
            activation='sigmoid',
            name='prediction'
        )(dense)

        model = Model(inputs=input_layer, ouputs=prediction)
        return model 

    def compile_model(self):
        """
        compile the initialized model
        """

        model = self.model
        model.compile(optimizer='adam', loss='binary_crossentropy')

        if self.verbose > 0:
            model.summary()

    def __post_init__(self):
        """
        after data initialization runs from the dataclass,
        initialize the model and also compile it.
        """

        self.model = self.get_model()
        self.compile_model()

    def __call__(self, x_train, y_train, x_test, y_test, epochs=100, batch_size=128):
        """
        fit the model around the given data

        :param x_train: the training data 
        :param y_train: the training labels 
        :param x_test: the testing data 
        :param y_test: the testing labels
        :param epochs: the number of epochs to fit for
        :param batch_size: the batch size to fit with 
        """
        history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=epochs,
            batch_size=batch_size
        )

        return history 


