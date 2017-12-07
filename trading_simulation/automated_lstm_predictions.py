#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import math
import matplotlib.pyplot as plt
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.layers import Embedding
# from keras.layers import LSTM

# TODO: follow the guide....
# helpful reference: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/


# Takes in Dataframes:
def pipeline_lstm(train_frame, labels_frame, cv_features_frame, cv_labels_frame):
    # Importing these within this context, so that they are not loaded every time the pipeline is run:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import Embedding
    from keras.layers import LSTM, ConvLSTM2D

    # print(train_frame)
    # print(labels_frame)

    train = train_frame.as_matrix()
    labels = labels_frame.as_matrix()

    cv_features = cv_features_frame.as_matrix()
    cv_labels = cv_labels_frame.as_matrix()

    print(train.shape)
    print(labels.shape)

    # Don't expect there to be timesteps... because features were extracted into timeseries, seperately...
    # Could explore how to build timesteps into this here...

    train = np.expand_dims(train, axis=1)
    cv_features = np.expand_dims(cv_features, axis=1)

    model = Sequential()
    # input_shape of LSTM first param is time steps...?
    model.add(LSTM(128, input_shape=train.shape[1:]))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(train, labels, batch_size=10, epochs=10, validation_data=(cv_features,cv_labels))
    print('LSTM Training Complete')
    score = model.evaluate(cv_features,cv_labels)

    print(score)


# A commandline runnable version:
def main():
    if len(sys.argv) < 3:
        print('Usage: ./automated_lstm_predictions.py extracted_features labels')
        exit(1)

    #Simply Would Need To Parse And Call The Above Process



if __name__ == '__main__':
	main()