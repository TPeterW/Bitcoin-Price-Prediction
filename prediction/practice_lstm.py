#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

def main():
    print('hi')

    size = 1350
    split = int(size * .6)
    filename = "/Users/HenrySwaffield/Documents/Middlebury Senior Year/fall/Senior Seminar/project/data/raw_data/btc-ohlc-coindesk.csv"
    file_data = pd.read_csv(filename, index_col=None, header=0, nrows=size)
    file_data = file_data[file_data.columns[1:]]

    matrix = pd.DataFrame.as_matrix(file_data)


    x_train = matrix[:,0]

    # print(x_train)

    y_train = matrix[:,3]

    # print(y_train)

    y_train = y_train > x_train

    # print(y_train)

    non_labels = matrix[:, :3]

    # x_min = np.min(non_labels)
    # range = np.range(non_labels)
    #
    # non_labels = non_labels - x_min
    # non_labels = non_labels / range
    # non_labels = non_labels * .9999999

    print(non_labels)

    max_features = 20000
    model = Sequential()
    model.add(Embedding(max_features, output_dim=1))
    input_shape = (810,1)
    # model.add(LSTM(1, input_shape=input_shape, return_sequences=False))
    model.add(LSTM(128))

    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


    # x_train = matrix[:, 1:4]
    # y_train = matrix[:, 4]
    # np.delete(x_train, size, 0)
    # np.delete(x_train, 0, 0)

    # print(x_train)
    # print(y_train)

    # print(file_data)


    # trying offsets...
    offset = 2
    x_train_set = non_labels[:split-offset]
    y_train_set = y_train[offset:split]
    print(x_train_set)
    print(y_train_set)

    model.fit(x_train_set, y_train_set, batch_size=10, epochs=20)
    # give the right test set here... labels might be off...
    score = model.evaluate(matrix[split:], y_train[split:], batch_size=16)
    print(score)

if __name__ == '__main__':
	main()