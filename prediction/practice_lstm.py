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

    size = 1300
    filename = "/Users/HenrySwaffield/Documents/Middlebury Senior Year/fall/Senior Seminar/project/data/raw_data/btc-ohlc-coindesk.csv"
    file_data = pd.read_csv(filename, index_col=None, header=0, nrows=size)

    matrix = pd.DataFrame.as_matrix(file_data)

    x_train = matrix[:,1]


    y_train = matrix[:,4]

    y_train = y_train > x_train

    x_min = min(x_train)
    range = max(x_train) - x_min;

    x_train = x_train - x_min
    x_train = x_train / range
    x_train = x_train * .99

    # print(x_train)

    max_features = 1
    model = Sequential()
    model.add(Embedding(max_features, output_dim=1))
    # input_shape = (1, size)
    model.add(LSTM(1))

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
    print(y_train)

    # print(file_data)
    

    model.fit(x_train, y_train, batch_size=16, epochs=10)
    # give the right test set here... labels might be off...
    score = model.evaluate(x_train, y_train, batch_size=16)
    print(score)

if __name__ == '__main__':
	main()