import sys
import itertools
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Takes in the training set of features and labels, and returns the trained model:
def train(features_train, labels_train):
	model = LogisticRegression(C=1e5)
	model.fit(features_train, labels_train)
	return model

# Takes in a trained model, and evaluates it with the test set:
# It also writes the predictions, to be used by downstream simulation processes
def test_model(trained_model, features_test, raw_price_data):
	predictions = trained_model.predict(features_test)

	bank = 0
	spent = 0
	coins = 0
	for index, pred in enumerate(predictions):
		# if not index % (60 * 24) == 0:
			# continue
		open_price = raw_price_data.iloc[index][0]
		close_price = raw_price_data.iloc[index][1]
		if pred == 0:
			if coins > 0:
				coins -= 1
				bank += close_price
		else:
			coins += 1
			spent += open_price
		
	net_earn = bank - spent + coins * close_price
	print('Assume infinite funds, total input is %s, total output is %s' % (str(spent), str(bank)))
	print('We have %s coins left in our account' % (str(coins)))
	print('And our total profit is %s' % (str(net_earn)))

# trains and tests a model with the provided train and test sets, writes predictions with the given output name
def train_and_simulate_process(features_train, labels_train, features_test, raw_price_data):
	model = train(features_train, labels_train)

	raw_price_data = raw_price_data[-len(features_test):].reset_index(drop=True)[['Open', 'Close']]
	test_model(model, features_test, raw_price_data)

def load_params():
	config = configparser.ConfigParser()
	config.read('../config.ini')

	global TRANSACTION_SIZE
	TRANSACTION_SIZE = int(config['Simulation']['base'])