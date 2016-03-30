from __future__ import absolute_import
from __future__ import print_function

import sys
import glob

import numpy as np
#import matplotlib.pyplot as plt

import pandas as pd
pd.set_option('display.width',    1000)
pd.set_option('display.max_rows', 1000)

from signals import *

def __main__():
    """ 
    Trading Simulator from curriculumvite trading competition
    see also the arvix Paper from Roni Mittelman http://arxiv.org/pdf/1508.00317v1
    Modified by Ernst.Tmp@gmx.at
    
    produces data to train a neural net
    """
    # Trades smaller than this will be omitted


    path = "./training_data_large/"
    file_list = sorted(glob.glob('./training_data_large/prod_data_*v.txt'))

    if len(file_list) == 0:
        print ("No ./training_data_large/product_data_*txt  files exist in the directory. Please copy them in the ./training_data_largetest/ . Aborting.")
        sys.exit()
        
    try:
        write_spans = True if sys.argv[1] == "--spans" else False
    except IndexError:
        write_spans = False
        
    try:
        #chained_deals = True if sys.argv[0] == "--chained-deals" else False
        chained_deals = True if sys.argv[1] == "--chained-deals" else False
    except IndexError:
        chained_deals = False    
    
    min_trade_amount = None
    comission = 0.0

    for j in range(len(file_list)):
        filename = file_list[j]
        print('Training: ',filename)

        day_file = filename
        
        generate_signals_for_file(day_file, comission, write_spans, chained_deals)


__main__();
