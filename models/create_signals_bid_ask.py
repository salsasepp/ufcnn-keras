from __future__ import absolute_import
from __future__ import print_function

import sys

import numpy as np
#import matplotlib.pyplot as plt

import pandas as pd
pd.set_option('display.width',    1000)
pd.set_option('display.max_rows', 1000)

def find_signals(df, sig_type, comission=0.0, debug=False):
    colnames = {"Buy": ("Buy", "Sell Close"),
                "Sell": ("Sell", "Buy Close")}
    inflection_points_buy = df["askpx_"].diff().shift(-1) > 0
    inflection_points_sell = df["bidpx_"].diff().shift(-1) < 0
    
    iterator = inflection_points_buy.iteritems() if sig_type == "Buy" else inflection_points_sell.iteritems()
    inflection_points = inflection_points_buy if sig_type == "Buy" else inflection_points_sell
    inner_inflection_points = inflection_points_sell if sig_type == "Buy" else inflection_points_buy
    
    max_count = 0
    
    (major_colname, minor_colname) = colnames[sig_type]
    
    df[major_colname] = np.zeros(df.shape[0])
    df[minor_colname] = np.zeros(df.shape[0])
    
    for idx, val in iterator:
        if max_count > 10000 and debug:
            print("Max count reached, break...")
            break
        inner_iterator = inner_inflection_points.loc[idx:].iteritems()
        if df[df[minor_colname]==1].empty:
            can_open = True
        else:
            can_open = idx > df[df[minor_colname]==1].index[-1]
        max_count += 1
        if val and can_open:
            print("{} candidate at {} with price {}".format(sig_type, idx, df["askpx_"][idx]))
            for inner_idx, inner_val in inner_iterator:
                if inner_idx > idx:
                    if sig_type == "Buy":
                        if df["askpx_"][inner_idx] < df["askpx_"][idx] and inflection_points[inner_idx]:
                            print("Better {} candidate at {} with price {}, break...".format(sig_type, inner_idx, df["askpx_"][inner_idx]))
                            break
                        if df["bidpx_"][inner_idx] > (df["askpx_"][idx] + comission) and inner_val:
                            df[major_colname][idx] = 1
                            df[minor_colname][inner_idx] = 1
                            print("Buy at {} with price {}".format(idx, df["askpx_"][idx]))
                            print("Sell at {} with price {}".format(inner_idx, df["bidpx_"][inner_idx]))
                            break
                    elif sig_type == "Sell":
                        if df["bidpx_"][inner_idx] > df["bidpx_"][idx] and inflection_points[inner_idx]:
                            print("Better {} candidate at {} with price {}, break...".format(sig_type, inner_idx, df["bidpx_"][inner_idx]))
                            break
                        if df["askpx_"][inner_idx] < (df["bidpx_"][idx] - comission) and inner_val:
                            df[major_colname][idx] = 1
                            df[minor_colname][inner_idx] = 1
                            print("Sell at {} with price {}".format(idx, df["bidpx_"][idx]))
                            print("Buy at {} with price {}".format(inner_idx, df["askpx_"][inner_idx]))
                            break   
    return df


def filter_signals(df):
    buys = df["Buy"] + df["Buy Close"]
    df["Buy"] = np.zeros(df.shape[0])
    df["Buy"][buys == 2] = 1
    sells = df["Sell"] + df["Sell Close"]
    df["Sell"] = np.zeros(df.shape[0])
    df["Sell"][sells == 2] = 1
    
    df = df.drop(["Buy Close", "Sell Close"], axis=1)
    return df


def make_spans(df, sig_type):
    span_colname = "Buys" if sig_type == "Buy" else "Sells"
    reversed_df = df[::-1]
    df[span_colname] = np.zeros(df.shape[0])
    
    for idx in df[sig_type][df[sig_type] == 1].index:
        signal_val = df.loc[idx]
        iterator = reversed_df.loc[idx:].iterrows()
        _d = print("Outer loop:", idx, signal_val["askpx_"]) if sig_type == "Buy" else print("Outer loop:", idx, signal_val["bidpx_"])
        for i, val in iterator:
            _d = print("Inner loop:", i, val["askpx_"]) if sig_type == "Buy" else print("Inner loop:", i, val["bidpx_"])
            if sig_type == "Buy":
                if val["askpx_"] == signal_val["askpx_"]:
                    print("Add to buys")
                    df[span_colname][i] = 1
                else:
                    break
            elif sig_type == "Sell":
                if val["bidpx_"] == signal_val["bidpx_"]:
                    print("Add to sells")
                    df[span_colname][i] = 1
                else:
                    break
    return df


def pnl(df):
    deals = []
    pnl = 0
    is_opened = False

    for idx, row in df.iterrows():
        if row["Buy"]:
            if is_opened:
                deals.append(-row["askpx_"])
            deals.append(-row["askpx_"])
            is_opened = True
        elif row["Sell"]:
            if is_opened:
                deals.append(row["bidpx_"])
            deals.append(row["bidpx_"])
            is_opened = True
    print(len(deals))
    deals.pop()
    print(len(deals))
    return np.sum(deals), len(deals)


def __main__():
    """ 
    Trading Simulator from curriculumvite trading competition
    see also the arvix Paper from Roni Mittelman http://arxiv.org/pdf/1508.00317v1
    Modified by Ernst.Tmp@gmx.at
    
    produces data to train a neural net
    """
    # Trades smaller than this will be omitted
    min_trade_amount = None
    comission = 0.0

    if len(sys.argv) < 2 :
        print ("Usage: day_trading_file, NOT target_price-file ")
        sys.exit()


    day_file = sys.argv[1]


    if "_2013" in day_file: 
        month = day_file.split("_")[2].split(".")[0]
        write_file = "signal_" + month + ".csv"
    else:
        write_file = "signal.csv"

    print("Processing file ",day_file)
    print("Writing to file ",write_file)

    df = pd.read_csv(day_file, sep=" ", usecols=[0,1,2,3,4,5], index_col = 0, header = None, names = ["time","mp","bidpx_","bidsz_","askpx_","asksz_",])
    df = find_signals(df, "Buy")
    df = find_signals(df, "Sell")
    df = filter_signals(df)
    df = make_spans(df, "Buy")    
    df = make_spans(df, "Sell")
    
    df['signal'] = np.zeros(df.shape[0])
    
    df['signal'][df["Buy"] == 1] = 1.0
    df['signal'][df["Sell"] == 1] = -1.0
    
    # and write the signal 
    signal_df = df[['signal', 'Buy', 'Sell', 'Buys', 'Sells']]
    signal_df.to_csv(write_file)

    #print ("Trades")
    #print(trades2_df)
    #print(trades3_df)
    #print ("Read DF from ", day_file)
    #print(df)
    _pnl, trade_count = pnl(df)
    print("Max. theoret. PNL    : ", _pnl) #df.sum().absret_)
    print("Max. theoret. return : ", _pnl / df["mp"].iloc[0])
    print("Max. number of trades: ", trade_count)
    print("Min Trading Amount   : ", min_trade_amount)


__main__();
