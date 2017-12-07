import sys
import os.path
from tsfresh import select_features
import pandas as pd

import automated_feature_extraction
import automated_predictions
from automated_lstm_predictions import pipeline_lstm

# value we have been using:
steps_look_back = 30
# TODO: connect to config file

global train_test_ratio

# Auto Extraction Script: In: raw data, minutes_look_back. Out: Extracted Features, Labels (perhaps to change)

# Auto Prediction Script: In: output of extraction and labeling, file names. Out: Predictions, for analysis

# Note:
# This is how you line up the raw_data and the predictions ... +/- 1
# raw_data_offset = minutes_before + cutoff


# reads the output from the labeling and feature extraction processes:
def read_feature_extraction_and_label_output(features, labels):
    features = pd.read_csv(features, index_col=None, header=0)
    labels = pd.read_csv(labels, index_col=None, header=None, squeeze=True)

    return features, labels


# This takes output from the feature extraction and labeling components:
# Reads csv... returns more convenient data_frames
def split_train_and_test(features, labels):
    global train_test_ratio

    split_point = int(len(features) * train_test_ratio)

    labels = labels[:len(features)].astype(int)
    features = select_features(features, labels)

    features_train = features.loc[:split_point]
    labels_train = labels.loc[:split_point]
    features_test = features.loc[split_point:].reset_index(drop=True)
    labels_test = labels.loc[split_point:].reset_index(drop=True)

    return features_train, labels_train, features_test, labels_test

# Parses the Input filename to produce a useful filestem.
def input_file_to_output_name(filename):
    get_base_file = os.path.basename(filename)
    base_filename = get_base_file.split('.')[0]
    # base_filename = '/pipeline_data/' + base_filename
    return base_filename

# Runs The whole pipeline:
def simulation_pipeline_process(raw_data_name, steps_look_back, model_choice):

    file_stem = input_file_to_output_name(raw_data_name)


    # Can check here to see if this stage is necessary:

    # Checking whether feature extraction work has been performed, with the provided input file
    # This will prevent that expensive procedure from occurring, if the pipeline already has the data it would compute
    if not os.path.isfile(file_stem + '_timeseries.csv') or not os.path.isfile(file_stem + '_labels.csv'):
        automated_feature_extraction.feature_extraction_process(raw_data_name, steps_look_back)
        print("Feature Extraction, Label Application, And Feature Selection Completed")
    else:
        print("Feature Extraction, Label Application, And Feature Selection Previously Completed")

    # Reading in the output from the labeling and feature extraction processes:
    features, labels = read_feature_extraction_and_label_output(file_stem + '_features_extracted.csv',
                                                                file_stem + '_labels.csv')
    features_train, labels_train, features_test, labels_test = split_train_and_test(features, labels)

    # Performing the model training and evaluation:

    #TODO: toggle the model:
    if model_choice:
        pipeline_lstm(features_train, labels_train, features_test, labels_test)
    else:
        automated_predictions.train_and_test_process(features_train, labels_train, features_test, labels_test, file_stem)


# Calls Pipeline Process
def main():
    if len(sys.argv) < 5:
        print('Usage: ./automated_simulation_pipeline.py raw_data train_test_ratio steps_look_back model_choice')
        exit(1)

    global train_test_ratio

    raw_data_name = sys.argv[1]
    train_test_ratio = float(sys.argv[2])
    steps_look_back = int(sys.argv[3])
    model_choice = bool(sys.argv[4])

    # param here to toggle the model used:
    simulation_pipeline_process(raw_data_name, steps_look_back, model_choice)


if __name__ == '__main__':
    main()
