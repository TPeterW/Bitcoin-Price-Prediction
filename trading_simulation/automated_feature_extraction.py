import sys
import os.path
import pandas as pd
from coinutils import input_file_to_output_name

from tqdm import tqdm
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

# def main():
#     print('running ', len(sys.argv))
#     if len(sys.argv) < 3:
#         print('Usage: ./automated_feature_extraction.py datafile.csv look_back_steps')
#         exit(1)

#     filename = sys.argv[1]
#     look_back_steps = int(sys.argv[2])

#     file_stem = input_file_to_output_name(filename)

#     if not os.path.isfile(file_stem + '_timeseries.csv') or not os.path.isfile(file_stem + '_labels.csv'):
#         feature_extraction_process(filename, look_back_steps)
#     else:
#         print("Intermediate Files Already Exist")

def convert(raw_price_data, look_back_steps):
    price_data = raw_price_data.astype(float)

    print('Generating labels...')
    close_prices = price_data['Close'].reset_index(drop=True)
    open_prices = price_data['Open'].reset_index(drop=True)

    labels = pd.Series([0] * len(price_data))
    for i in tqdm(range(len(price_data) - 1, look_back_steps - 1, -1)):
        if close_prices[i] > open_prices[i]:
            labels[i] = 1
        else:
            labels[i] = 0
    labels = labels.reset_index(drop=True)[look_back_steps:]

    print('Removing redundent columns...')
    for col in price_data.columns:
        if 'high' in col.lower() or 'low' in col.lower() or 'close' in col.lower():
            price_data.drop(col, axis=1, inplace=True)

    print('Converting into timeseries...')
    raw = []
    for i in tqdm(range(look_back_steps, len(price_data))):
        for j in range(look_back_steps):
            row = price_data.loc[i - look_back_steps + j].tolist()
            raw.append([i - look_back_steps, j] + row)

    timeseries = pd.DataFrame(raw, index=None, columns=['id', 'time'] + price_data.columns.tolist())

    return timeseries, labels


# Called by the pipeline
def feature_extraction_process(filename, look_back_steps):
    base_filename = input_file_to_output_name(filename)
    raw_price_data = pd.read_csv(filename, index_col=None, header=0, thousands=',')

    timeseries, labels = convert(raw_price_data, look_back_steps)

    features = extract_features(timeseries, column_id='id', column_sort='time')
    impute(features)
    features.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)

    timeseries.to_csv(base_filename + '_timeseries.csv', index=False, header=True)
    features.to_csv(base_filename + '_features_extracted.csv', sep=',', index=False, header=True)
    labels.to_csv(base_filename + '_labels.csv', sep=',', index=False, header=False)

    print('Output Labeling, and Feature Extraction Completed')

# if __name__ == '__main__':
#     main()
