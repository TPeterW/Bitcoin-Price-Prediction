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
    size = 30485
    # size = 200000
    split = int(size * .6)

    # Coinbase data set is sparse before row 418053... there are no changes, so it is useles...
    useless_rows = 418053

    # When the split is .4 max accuracy is 52% on test set
    # when split is .6 max accuracy is 59% on test set
    # when split is .8 max accuracy is 62.83% on test set

    # original input type ^^

    # new input type, in the custom feature, simply look if next day percent change is positive:
    # Maybe this was on the bad initial data...
    # It's looking likt that was the case...
    # split .8 Epoch 2/20
    # 1080/1080 [==============================] - 2s - loss: 0.6941 - acc: 0.4843 - val_loss: 0.6880 - val_acc: 0.6407
    # Epoch 3/20
    # 1080/1080 [==============================] - 2s - loss: 0.6938 - acc: 0.5176 - val_loss: 0.6823 - val_acc: 0.6407
    # Epoch 4/20
    # 1080/1080 [==============================] - 2s - loss: 0.6891 - acc: 0.5352 - val_loss: 0.7120 - val_acc: 0.4259

    # with split at .6 we get .57% accuracy on the new feature being tested

    offset = 0

    # filename = "/Users/HenrySwaffield/Documents/Middlebury Senior Year/fall/Senior Seminar/project/data/raw_data/coinbaseUSD_1-min_data_2014-12-01_to_2017-10-20.csv_test2.csv"
    filename = "/Users/HenrySwaffield/Documents/Middlebury Senior Year/fall/Senior Seminar/project/data/raw_data/coinbaseUSD_1-min_data_test3.csv"


    file_data = pd.read_csv(filename, skiprows=useless_rows, index_col=None, header=0, nrows=size)
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

    # non_labels = matrix[:, 1:12]
    non_labels = matrix[:, 1:14]

    # One minute later guesses, as labels:
    # y_train = matrix[:, 12:13]
    # This here is checking if the close of the next minute will be greater than the close of the current minute...
    y_train = matrix[:, 14:15]
    # Getting about 61% accuracy... not that much better than before...
    # 62910 / 62910[ == == == == == == == == == == == == == == ==] - 301
    # s - loss: 0.6583 - acc: 0.6119 - val_loss: 0.6639 - val_acc: 0.5937
    # Epoch
    # 2 / 20
    # 62910 / 62910[ == == == == == == == == == == == == == == ==] - 312
    # s - loss: 0.6550 - acc: 0.6122 - val_loss: 0.6630 - val_acc: 0.5961


    # most up to date results:

    # 62910 / 62910[ == == == == == == == == == == == == == == ==] - 314
    # s - loss: 0.6580 - acc: 0.6116 - val_loss: 0.6679 - val_acc: 0.6070
    # Epoch
    # 2 / 5
    # 62910 / 62910[ == == == == == == == == == == == == == == ==] - 318
    # s - loss: 0.6547 - acc: 0.6116 - val_loss: 0.6733 - val_acc: 0.5493
    # Epoch
    # 3 / 5
    # 62910 / 62910[ == == == == == == == == == == == == == == ==] - 321
    # s - loss: 0.6510 - acc: 0.6215 - val_loss: 0.6570 - val_acc: 0.6150
    # Epoch
    # 4 / 5
    # 62910 / 62910[ == == == == == == == == == == == == == == ==] - 322
    # s - loss: 0.6120 - acc: 0.6764 - val_loss: 0.6454 - val_acc: 0.6372
    # Epoch
    # 5 / 5
    # 62910 / 62910[ == == == == == == == == == == == == == == ==] - 323
    # s - loss: 0.6108 - acc: 0.6830 - val_loss: 0.6439 - val_acc: 0.6359


    # represents the 10 minute later guesses...
    # Gets around 55% accuracy...
    # y_train = matrix[:, 8:9]

    # 6291 / 6291[ == == == == == == == == == == == == == == ==] - 28
    # s - loss: 0.6885 - acc: 0.5498 - val_loss: 0.6898 - val_acc: 0.5510
    # Epoch
    # 2 / 20
    # 6291 / 6291[ == == == == == == == == == == == == == == ==] - 28
    # s - loss: 0.6801 - acc: 0.5680 - val_loss: 0.6973 - val_acc: 0.5033
    # Epoch
    # 3 / 20
    # 6291 / 6291[ == == == == == == == == == == == == == == ==] - 28
    # s - loss: 0.6764 - acc: 0.5811 - val_loss: 0.6936 - val_acc: 0.5346
    # Epoch
    # 4 / 20
    # 6291 / 6291[ == == == == == == == == == == == == == == ==] - 29
    # s - loss: 0.6741 - acc: 0.5800 - val_loss: 0.7104 - val_acc: 0.4700
    # Epoch
    # 5 / 20
    # 6291 / 6291[ == == == == == == == == == == == == == == ==] - 30
    # s - loss: 0.6708 - acc: 0.5912 - val_loss: 0.7034 - val_acc: 0.4731
    # Epoch
    # 6 / 20
    # Overfitting seems to be an issue...

    # average over the next 20 minutes being higher than current close...:
    # this does not work well, with smaller data set...
    # y_train = matrix[:, 9:10]
    # y_train = matrix[:, 10:11]  # random labels
    # y_train = matrix[:, 11:12]

    print('y_train', y_train)

    # x_min = np.min(non_labels)
    # range = np.range(non_labels)
    #
    # non_labels = non_labels - x_min
    # non_labels = non_labels / range
    # non_labels = non_labels * .9999999

    print('non_labels', non_labels)

    # Need to consider how this number of features could affect overfitting...
    max_features = 400000
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

    # bigger batch sizes makes traning much faster.
    model.fit(x_train_set, y_train_set, batch_size=50, epochs=20, validation_data=(x,y))
    # give the right test set here... labels might be off...
    score = model.evaluate(non_labels[split:size-offset], y_train[split:], batch_size=2)
    print(score)

if __name__ == '__main__':
	main()