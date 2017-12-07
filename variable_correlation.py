import sys
import pandas as pd
import numpy as np
import csv

# Runs an analysis to see correlations between variables
def run_correlations(filename):
    data = pd.read_csv(filename, index_col=None, header=0)

    alt_coins = ["Litecoin", "Namecoin", "Novacoin", "Nxt", "Peercoin", "Primecoin", "Quark", "Ripple",
                 "TagCoin", "Unobtanium", "WorldCoin", "Zetacoin"]

    btc_o = data['Open']
    btc_c = data['Close']
    btc_change = (btc_c - btc_o) / btc_o

    ltc_o = data['Litecoin-Open']
    ltc_c = data['Litecoin-Close']

    ltc_change = (ltc_c - ltc_o) / ltc_o

    for coin in alt_coins:
        coin_open = data[coin+'-Open']
        coin_close = data[coin + '-Close']
        coin_change = (coin_close - coin_open) / coin_open

        coefficient_same_day = np.corrcoef(coin_change, btc_change)[0, 1]

        num_steps = len(btc_change) - 1

        coefficient_next_day_btc = np.corrcoef(coin_change[:num_steps], btc_change[1:])[0, 1]

        print("Testing Correlation Between Bitcoin Historical Data and (same day %change) " + coin + ": ")
        print("R (same day)= " + str(coefficient_same_day))
        print("R (next day)= " + str(coefficient_next_day_btc))





def main():
    if len(sys.argv) < 2:
        print('Usage: ./variable_correlation consolidated_coins.csv')
        exit(1)

    # filename = sys.argv(1)
    filename = 'consolidated_data_with_coins_newest.csv'
    run_correlations(filename)




if __name__ == '__main__':
	main()