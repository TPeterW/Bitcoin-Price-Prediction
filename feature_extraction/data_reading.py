#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd

def read_files(data_files):
    print('hi')

    # Where input data will live:
    data = {}

    for filename in data_files:
        print(filename)
        file_data = pd.read_csv(filename, index_col=None, header=0, squeeze=True, usecols=[1], skiprows=[0])
        data[filename] = file_data
    return data
