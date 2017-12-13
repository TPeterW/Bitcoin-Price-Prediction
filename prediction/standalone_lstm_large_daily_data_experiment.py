from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
set_random_seed(3)
# For making reproducible results ^, partially works.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, BatchNormalization

def main():

    # 4 years of daily data:
    size = 1369
    split = int(size * .8)

    filename = '/Users/HenrySwaffield/Documents/Middlebury Senior Year/fall/Senior Seminar/project/workspace/Bitcoin-Price-Prediction/data_collection/consolidated_data_with_coins_newest.csv'

    file_data = pd.read_csv(filename, index_col=None, header=0, nrows=size)
    matrix = pd.DataFrame.as_matrix(file_data)

    # There are 114 input fields:
    non_labels = matrix[:, 0:114]
    # Label Fields:
    labels = matrix[:, 114:115]

    # trying offsets...
    train = np.expand_dims(non_labels, axis=1)

    x_train_set = train[:split]
    y_train_set = labels[0:split]

    x = train[split:size]
    y = labels[split:]


    # Adding dropout here increased accuracy a lot:
    model = Sequential()
    model.add(BatchNormalization(input_shape=train.shape[1:]))
    # input_shape of LSTM first param is time steps...?
    model.add(LSTM(115,  input_shape=train.shape[1:], return_sequences=True))
    model.add(Dropout(.2))
    model.add(LSTM(115, return_sequences=True))
    model.add(Dropout(.2))
    model.add(LSTM(115))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print(model.summary())

    # bigger batch sizes makes traning much faster.
    history = model.fit(x_train_set, y_train_set, batch_size=5, epochs=20, validation_data=(x,y))
    print('[loss, accuracy]:')
    score = model.evaluate(x, y, batch_size=200)
    print(score)

    # for plotting learning metrics curves:
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
	main()
