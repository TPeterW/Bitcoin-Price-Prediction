import sys
import os.path
from tsfresh import select_features

import automated_feature_extraction
import automated_predictions

# Steps:

# ? where to derive labels?? does it make sense to do it in feature extraction??

steps_look_back = 30
raw_data = '/Users/HenrySwaffield/Documents/Middlebury Senior Year/fall/Senior Seminar/project/workspace/Bitcoin-Price-Prediction/prediction/input_test1.csv'

global train_test_ratio

# Auto Extraction Script: In: raw data, minutes_look_back. Out: Extracted Features, Labels (perhaps to change)


# Make A New Prediction Script:
#

# These should all take the appropriate lookback params, cutoff, etc...


# This will manage the raw_data
# Split train and test data here...

# This is how you line up the raw_data and the predictions ... +/- 1
# raw_data_offset = minutes_before + cutoff

# manage descriptive name here...


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


def input_file_to_output_name(filename):
    get_base_file = os.path.basename(filename)
    base_filename = get_base_file.split('.')[0]
    # base_filename = '/pipeline_data/' + base_filename
    return base_filename


# maybe more params to specify pipeline options:
# TODO: make a seperate pipeline function
def main():
    global train_test_ratio

    raw_data_name = sys.argv[1]

    train_test_ratio = float(sys.argv[2])

    steps_look_back = int(sys.argv[3])

    # Can check here to see if this stage is necessary:
    automated_feature_extraction.feature_extraction_process(raw_data_name, steps_look_back)




if __name__ == '__main__':
    main()