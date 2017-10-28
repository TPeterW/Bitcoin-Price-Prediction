#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import csv

num_fields = 1
fieldnames = ['date']
num_rows = 1369
# should be 1369, lower because some coins have missing days...
output_name = "consolidated_data_with_coins.csv"


def main():
    if len(sys.argv) >= 2:
        global output_name
        output_name = sys.argv[1]

    # just for testing on my own computer, for now: use ^^^ normally
    filepath = "/Users/HenrySwaffield/Documents/Middlebury Senior Year/fall/Senior Seminar/project/data/raw_data/"
    # filepath will change... provide th appropriate path for your system ... this could synchonize with google docs.

    filenames = [filepath + "btc-ohlc-coindesk.csv", filepath + "n-daily-btc-transactions.csv", filepath + "num-btc-transactions-per-block.csv", filepath + "btc-estimated-transaction-volume-blockchain-in-usd.csv"]
    filenames.append(filepath + "fee-cost-per-transaction-percent.csv")

    # coin_csvs = ["Anoncoin_flipped.csv", "Argentum_flipped.csv", "BBQCoin_flipped.csv", "BetaCoin_flipped.csv", "BitBar_flipped.csv", "Bitcoin_flipped.csv", "BitShares_flipped.csv", "CasinoCoin_flipped.csv", "Catcoin_flipped.csv", "Copperlark_flipped.csv", "CraftCoin_flipped.csv", "Datacoin_flipped.csv", "Devcoin_flipped.csv", "Diamond_flipped.csv", "Digitalcoin_flipped.csv", "Dogecoin_flipped.csv", "EarthCoin_flipped.csv", "Elacoin_flipped.csv", "EZCoin_flipped.csv", "Fastcoin_flipped.csv", "Feathercoin_flipped.csv", "FlorinCoin_flipped.csv", "Franko_flipped.csv", "Freicoin_flipped.csv", "GameCoin_flipped.csv", "GlobalCoin_flipped.csv", "GoldCoin_flipped.csv", "GrandCoin_flipped.csv", "HoboNickels_flipped.csv", "I0Coin_flipped.csv", "Infinitecoin_flipped.csv", "Ixcoin_flipped.csv", "Joulecoin_flipped.csv", "Litecoin_flipped.csv", "LottoCoin_flipped.csv", "Luckycoin_flipped.csv", "Mastercoin_flipped.csv", "Megacoin_flipped.csv", "Memorycoin_flipped.csv", "Mincoin_flipped.csv", "Namecoin_flipped.csv", "NetCoin_flipped.csv", "Noirbits_flipped.csv", "Novacoin_flipped.csv", "Nxt_flipped.csv", "Orbitcoin_flipped.csv", "Peercoin_flipped.csv", "Phoenixcoin_flipped.csv", "Primecoin_flipped.csv", "Quark_flipped.csv", "Ripple_flipped.csv", "Spots_flipped.csv", "StableCoin_flipped.csv", "TagCoin_flipped.csv", "Terracoin_flipped.csv", "Tickets_flipped.csv", "Tigercoin_flipped.csv", "Unobtanium_flipped.csv", "WorldCoin_flipped.csv", "Yacoin_flipped.csv", "Zetacoin_flipped.csv"]
    # ^^ some of the data has missing days...

    coin_csvs = ["Litecoin_flipped.csv", "Namecoin_flipped.csv", "Novacoin_flipped.csv", "Nxt_flipped.csv", "Peercoin_flipped.csv", "Primecoin_flipped.csv", "Quark_flipped.csv", "Ripple_flipped.csv", "StableCoin_flipped.csv", "TagCoin_flipped.csv", "Terracoin_flipped.csv", "Tickets_flipped.csv", "Tigercoin_flipped.csv", "Unobtanium_flipped.csv", "WorldCoin_flipped.csv", "Yacoin_flipped.csv", "Zetacoin_flipped.csv"]

    coin_path = filepath+"historical_coins_flipped/"

    coin_csvs = [coin_path + s for s in coin_csvs]

    filenames.extend(coin_csvs)
    # also subject to change^ these are the datasets that I have cleaned.

    write_csv(read_files(filenames))


def read_files(data_files):
    global fieldnames
    global num_fields

    # Structures to hold the data as it is read in:
    dates = pd.read_csv(data_files[0], index_col=None, header=0, nrows=num_rows)
    dates = dates[dates.columns[:1]]
    data_matrix = pd.DataFrame.as_matrix(dates)

    for filename in data_files:
        print(filename)
        coin = filename.split('/')[-1].split('.')[0].split('_')[0] + '-'
        print(coin)
        # data_name = filename.split('/')[-1].split('.')[0]
        # TODO: Look out for data where there is randomly new data frequencies... sometimes it changes...
        file_data = pd.read_csv(filename, index_col=None, header=0, nrows=num_rows)
        file_data = file_data[file_data.columns[1:]]

        file_matrix = pd.DataFrame.as_matrix(file_data)

        print(file_matrix.shape)

        # print(file_matrix)
        if file_matrix.shape[0] == 1369:
            data_matrix = np.hstack((data_matrix, file_matrix))
            cols = list(file_data.columns)

            cols = [coin + col for col in cols]

            fieldnames.extend(cols)
            num_fields += len(cols)

    return data_matrix


def write_csv(data):
    with open(output_name, 'w') as csvfile:
        # w is write only, and it will overwrite an existing file (truncates to 0 length)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for i in range(num_rows):
            current_row = {}
            for j in range(num_fields):
                current_row[fieldnames[j]] = data[i][j]
            writer.writerow(current_row)


if __name__ == '__main__':
	main()