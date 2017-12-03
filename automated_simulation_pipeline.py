
import os.path

# Steps:

# ? where to derive labels?? does it make sense to do it in feature extraction??

steps_look_back = 30
raw_data = '/Users/HenrySwaffield/Documents/Middlebury Senior Year/fall/Senior Seminar/project/workspace/Bitcoin-Price-Prediction/prediction/input_test1.csv'

# Auto Extraction Script: In: raw data, minutes_look_back. Out: Extracted Features, Labels (perhaps to change)


# Make A New Prediction Script:
#

# These should all take the appropriate lookback params, cutoff, etc...


# This will manage the raw_data
# Split train and test data here...

# This is how you line up the raw_data and the predictions ... +/- 1
# raw_data_offset = minutes_before + cutoff

# manage descriptive name here...


def input_file_to_output_name(filename):
    get_base_file = os.path.basename(filename)
    base_filename = get_base_file.split('.')[0]
    # base_filename = '/pipeline_data/' + base_filename
    return base_filename
