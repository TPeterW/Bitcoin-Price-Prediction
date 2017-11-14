#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import pandas as pd


def main():
    if len(sys.argv) >= 2:
        file_name = sys.argv[1]
        flipfile(file_name)

    # else:
    #     files = ["Anoncoin.csv", "Argentum.csv", "BBQCoin.csv", "BetaCoin.csv", "BitBar.csv", "Bitcoin.csv",
    #              "BitShares.csv", "CasinoCoin.csv", "Catcoin.csv", "Copperlark.csv", "CraftCoin.csv", "Datacoin.csv",
    #              "Devcoin.csv", "Diamond.csv", "Digitalcoin.csv", "Dogecoin.csv", "EarthCoin.csv", "Elacoin.csv",
    #              "EZCoin.csv", "Fastcoin.csv", "Feathercoin.csv", "FlorinCoin.csv", "Franko.csv", "Freicoin.csv",
    #              "GameCoin.csv", "GlobalCoin.csv", "GoldCoin.csv", "GrandCoin.csv", "HoboNickels.csv", "I0Coin.csv",
    #              "Infinitecoin.csv", "Ixcoin.csv", "Joulecoin.csv", "Litecoin.csv", "LottoCoin.csv", "Luckycoin.csv",
    #              "Mastercoin.csv", "Megacoin.csv", "Memorycoin.csv", "Mincoin.csv", "Namecoin.csv", "NetCoin.csv",
    #              "Noirbits.csv", "Novacoin.csv", "Nxt.csv", "Orbitcoin.csv", "Peercoin.csv", "Phoenixcoin.csv",
    #              "Primecoin.csv", "Quark.csv", "Ripple.csv", "Spots.csv", "StableCoin.csv", "TagCoin.csv", "Terracoin.csv",
    #              "Tickets.csv", "Tigercoin.csv", "Unobtanium.csv", "WorldCoin.csv", "Yacoin.csv", "Zetacoin.csv"]
    #
    #     for filename in files:
    #         flipfile(filename)


def flipfile(filename):
    data = pd.DataFrame.from_csv(filename, index_col=None, header=0)
    data = data.iloc[::-1]
    data.to_csv(filename[:-4] + '_flipped.csv', index=False, header=True)


if __name__ == '__main__':
    main()