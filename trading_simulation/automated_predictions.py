import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# def main():
#     if len(sys.argv) < 3:
#         print('Usage: ./automated_predictions.py features labels raw_data_name')
#         exit(1)

#     features = pd.read_csv(sys.argv[1], index_col=None, header=0)
#     labels = pd.read_csv(sys.argv[2], index_col=None, header=None, squeeze=True)
#     raw_data_name = sys.argv[3]

#     features_train, labels_train, features_test, labels_test = automated_simulation_pipeline.read_feature_extraction_and_label_output(features, labels)

#     descriptive_output_name = automated_simulation_pipeline.input_file_to_output_name(raw_data_name)

#     train_and_test_process(features_train, labels_train, features_test, labels_test, descriptive_output_name)


# Takes in the training set of features and labels, and returns the trained model:
def train(features_train, labels_train):
    model = LogisticRegression(C=1e5)
    model.fit(features_train, labels_train)
    return model

# Takes in a trained model, and evaluates it with the test set:
# It also writes the predictions, to be used by downstream simulation processes
def test_model(trained_model, features_test, labels_test, descriptive_output_name):
    predictions = trained_model.predict(features_test)
    # score = trained_model.score(features_test, labels_test)

    np.savetxt(descriptive_output_name + '_predictions.csv', predictions, fmt='%i', delimiter=',')

    print('Accuracy: %s' % accuracy_score(labels_test, predictions))
    print('Precision: %s' % precision_score(labels_test, predictions))
    print('Recall: %s' % recall_score(labels_test, predictions))

    # Does not make sense to show this in this part of the code:
    # cnf_matrix = confusion_matrix(labels_test, predictions, [0, 1])
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, normalize=True, classes=[0, 1], title='Nomalized Confusion Matrix')
    # plt.show()

# trains and tests a model with the provided train and test sets, writes predictions with the given output name
def train_and_test_process(features_train, labels_train, features_test, labels_test, descriptive_output_name):
    model = train(features_train, labels_train)
    test_model(model, features_test, labels_test, descriptive_output_name)

# prints the confusion matrix, when enabled:
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# if __name__ == '__main__':
#     main()