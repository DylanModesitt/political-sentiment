import numpy as np
from keras.models import Sequential, Model 
from keras.layers import LSTM, Input, Dropout, Bidirectional, Dense


X_train = np.random.rand(430,600,6)
Y_train = np.random.rand(430,10)

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = ( 
    X_train.shape[1], 6)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


input_layer = Input(shape=(600,6,))

lstm = Bidirectional(
    LSTM(250), 
    merge_mode='concat'
)(input_layer)

pred = Dense(10)(lstm)
model = Model(inputs=input_layer, outputs=pred)
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train, Y_train, epochs = 20, batch_size = 32)




