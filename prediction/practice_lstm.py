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

    # size = 1048576
    size = 10485
    # size = 200000
    split = int(size * .6)

    # When the split is .4 max accuracy is 52% on test set
    # when split is .6 max accuracy is 59% on test set
    # when split is .8 max accuracy is 62.83% on test set

    # original input type ^^

    # new input type, in the custom feature, simply look if next day percent change is positive:
    # split .8 Epoch 2/20
    # 1080/1080 [==============================] - 2s - loss: 0.6941 - acc: 0.4843 - val_loss: 0.6880 - val_acc: 0.6407
    # Epoch 3/20
    # 1080/1080 [==============================] - 2s - loss: 0.6938 - acc: 0.5176 - val_loss: 0.6823 - val_acc: 0.6407
    # Epoch 4/20
    # 1080/1080 [==============================] - 2s - loss: 0.6891 - acc: 0.5352 - val_loss: 0.7120 - val_acc: 0.4259

    # with split at .6 we get .57% accuracy on the new feature being tested

    offset = 0

    filename = "/Users/HenrySwaffield/Documents/Middlebury Senior Year/fall/Senior Seminar/project/data/raw_data/custom_minute_test1.csv"

    file_data = pd.read_csv(filename, index_col=None, header=0, nrows=size)
    file_data = file_data[file_data.columns[1:]]

    matrix = pd.DataFrame.as_matrix(file_data)

    x_train = matrix[:,0]

    # offsets:

    x_train = x_train[:size - offset]

    # print(x_train)

    # y_train = matrix[:,5]
    #
    # y_train = y_train[offset : ]
    #
    # print(y_train)

    # y_train = y_train = x_train

    # print(y_train)

    # non_labels = matrix[:, :5]

    non_labels = matrix[:, 1:7]

    y_train = matrix[:, 7:]

    print('y_train', y_train)

    # x_min = np.min(non_labels)
    # range = np.range(non_labels)
    #
    # non_labels = non_labels - x_min
    # non_labels = non_labels / range
    # non_labels = non_labels * .9999999

    print('non_labels', non_labels)

    max_features = 2000000
    model = Sequential()
    model.add(Embedding(max_features, output_dim=1))
    input_shape = (810,1)
    # model.add(LSTM(1, input_shape=input_shape, return_sequences=False))

    # model.add(LSTM(128))

    model.add(LSTM(128))

    model.add(Dropout(0.5))
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

    x_train_set = non_labels[:split-offset]
    y_train_set = y_train[0:split - offset]
    print(x_train_set)
    print(y_train_set)

    x = non_labels[split:size-offset]
    y = y_train[split:]

    model.fit(x_train_set, y_train_set, batch_size=5, epochs=20, validation_data=(x,y))
    # give the right test set here... labels might be off...
    score = model.evaluate(non_labels[split:size-offset], y_train[split:], batch_size=2)
    print(score)

if __name__ == '__main__':
	main()