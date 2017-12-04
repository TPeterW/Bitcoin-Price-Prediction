#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import csv


def main():
    if len(sys.argv) < 3:
        print('Usage: ./trading_simulation.py raw_data predictions cutoff')
        exit(1)

# TODO:    implement various trading strategies based on predictions and the actual data...
# cutoff determines where the test data starts... predictions only correspond to those days...
# maybe that is why it would be good to pass the raw data into predict as well.

# in the pipeline, which could have numerous components / params, differing labels could correspond to different strategies.



# what this file needs to know:
# original price data source
# where the cutoff is between train and test, and what the lookback window is...
# the predictions


# THE PIPELINE is happening here...

if __name__ == '__main__':
	main()