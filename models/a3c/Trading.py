
import json
import numpy as np
np.set_printoptions(threshold=np.inf)
import random

import DataStore

from constants import TRADING_FEE



class Trading(object):

    def __init__(self, data_store=None, sequence_length=500, features_length=32, testing=False ):

     
        self.data_store = data_store
        self.training_days = data_store.get_number_days()
        self.sequence_length = sequence_length

        self.position_store = {}
        self.initrate_store = {}
        self.features_length = features_length
        # Will be broadcasted to the other Modules. Change if changing actions below...
        self.action_count = 3
        self.trading_fee = 0.2
        self.testing = testing
        self.iday = -1 # for testing

    def reset(self):

        # randomize days
        if not self.testing: 
            self.iday = np.random.randint(0, self.training_days)
        else:
            self.iday += 1 # iterate over all idays
            if self.iday > self.data_store.get_number_days():
                self.iday = 0

        self.day_length = self.data_store.get_day_length(self.iday)

        self.current_index = self.sequence_length 

        self.position = 0.
        self.initrate = 0.
        print ("IDAY:" , self.iday, self.current_index)
   
        self.current_rate_bid_norm, self.current_rate_bid, self.current_rate_ask_norm, self.current_rate_ask = self.data_store.get_bid_ask(self.iday, self.current_index)
        self.daily_reward = 0.
        self.last_profit = 0.
        self.daily_trades = 0
        self.daily_long_trades = 0
        self.daily_short_trades = 0
        self.daily_wins = 0.

    def get_reward(self, action):
        """ #reward, terminal, self._screen = get_reward(action) # Screen is 84 x 8 i
        This is the version without sliding
        added a term in get_reward that adds winning trades twice to the reward - 
           once each tick it lasts (here winning and losing trades are treated equally), 
           and when closing a winning trade, the winning amount is credited again. 
           So reward != pnl. See daily_pnl instead.
        """
        reward = 0 
        debug = False
     
        #store the last rates...
        last_rate_bid = self.current_rate_bid
        last_rate_ask = self.current_rate_ask
        terminal = False
        self.new_trade = False
        close_trade = False
 
        OFFSET = 100

        if self.current_index >= self.day_length - 2 - OFFSET:
            if abs(self.position) > 0.1:
                close_trade = True
            terminal = True
            self.position = 0 
        else:
            if action == 0 or action == 1112: # STAY/GO_SHORT
                if self.position >- 0.1:
                    if self.position > 0.1:
                        close_trade = True
                    if debug:
                        print("Going SHORT: ",index, self.current_rate_bid)
                    self.initrate = self.current_rate_bid # SELL at the BID
                    initrate_norm = self.current_rate_bid_norm # SELL at the BID
                    self.position = -1 # only 1 contract 
                    self.new_trade = True
                    reward -= TRADING_FEE * abs(self.position)
                    last_rate_ask = self.initrate # for correct & simple position calculation
                    self.daily_trades += 1
                    self.daily_short_trades += 1

            if action == 1: # STAY/GO_LONG 
                #if self.position == 0:
                if self.position < 0.1:
                    if self.position < -0.1:
                        close_trade = True
                    if debug:
                        print("Going LONG: ",index, self.current_rate_bid)
                    self.initrate = self.current_rate_ask # BUY at the ASK
                    initrate_norm = self.current_rate_ask_norm # 
                    self.position = 1 # only 1 contract 
                    self.new_trade = True
                    reward -= TRADING_FEE * abs(self.position)
                    last_rate_bid = self.initrate # for correct & simple position calculation
                    self.daily_trades += 1
                    self.daily_long_trades += 1

            if action == 2: # STAY/GO_FLAT
                if abs(self.position) == 1: ## make useless
                    if debug:
                        print("Going FLAT: ",index, self.current_rate_bid)
                    self.position = 0 
                    close_trade = True

        # move to the next time step...
        self.current_index += 1

        # enforce a minimum holding period
        if self.new_trade:
            self.current_index += OFFSET


        # and get the rates...
        self.current_rate_bid_norm, self.current_rate_bid, self.current_rate_ask_norm, self.current_rate_ask = self.data_store.get_bid_ask(self.iday, self.current_index)

        value = 0.

        # LONG: CURRENT_BID - INITIAL_ASK
        if self.position > 0.1:
            value = self.position * (self.current_rate_bid - last_rate_bid)

        # SHORT: CURRENT_ASK - INITIAL_BID
        if self.position < -0.1:
            value = self.position * (self.current_rate_ask - last_rate_ask) 


        # get the reward
        reward += value
        self.daily_reward += reward
        self.last_profit += reward
        if close_trade and self.last_profit > 0:
            reward += self.last_profit
            self.daily_wins += self.last_profit
            self.last_profit = 0.
            
        # TODO Need to redesign reward function to penalize bad trades correctly?
        
        # TODO test reward clipping?

        # integer 0-255 format 84 * 84
        # screen 84 * 84 = 7056
        # 250 * 4 = 1000
        # 1764 * 4= 1763; sinus 1763 + 6000 lang, sequence = 1764 als kopie, die ersten 4 bytes auf 0 setzen, dort die positionm reinschreiben und den einstandskurs.
        # sequence mit der laenge aus dem Data Store, dann die position und den init rate norm drauf schlagen in position 1763.
        # resizen auf 84 * 84.

        inputs = self.data_store.get_sequence(self.iday, self.current_index).copy()
  
        inputs[0,0] = self.position

        screen = np.resize(inputs, (84,8))
        if terminal:
            print ("Daily: index/reward$/win$/short/long/", self.current_index, self.daily_reward, self.daily_wins, self.daily_short_trades, self.daily_long_trades)

        return reward, terminal, screen # Screen is 84 x 84 i

      
    def get_reward_sliding(self, action):
        """ #reward, terminal, self._screen = get_reward(action) # Screen is 84 x 84 i
        This is the sliding version that slides between short and long positions...
        0) Terminal? 
        1) Action auswerten
        2) T = T + 1 
        3) Reward = ...
        4) get the next sequence...
        """
        reward = 0 
        debug = False
     
        #store the last rates...
        last_rate_bid = self.current_rate_bid
        last_rate_ask = self.current_rate_ask
        terminal = False

        if self.current_index == self.day_length - 2:
            terminal = True
            self.position = 0 
            print ("Daily Reward ", self.daily_reward)
        else:
            if action == 0: # STAY/GO_SHORT
                if self.position == 1:
                    self.position -= 1
                else: 
                    if self.position == 0:
                        if debug:
                            print("Going SHORT: ",index, self.current_rate_bid)
                        self.initrate = self.current_rate_bid # SELL at the BID
                        initrate_norm = self.current_rate_bid_norm # SELL at the BID
                        self.position = -1 # only 1 contract 
                        self.new_trade = True
                        reward -= TRADING_FEE * abs(self.position)
                        last_rate_ask = self.initrate # for correct & simple position calculation

            if action == 1: # STAY/GO_LONG 
                #if self.position == 0:
                if self.position == -1:
                    self.position += 1
                else: 
                    if self.position < 0.1:
                        if debug:
                            print("Going LONG: ",index, self.current_rate_bid)
                        self.initrate = self.current_rate_ask # BUY at the ASK
                        initrate_norm = self.current_rate_ask_norm # 
                        self.position = 1 # only 1 contract 
                        self.new_trade = True
                        reward -= TRADING_FEE * abs(self.position)
                        last_rate_bid = self.initrate # for correct & simple position calculation

            if action == 2: # Do nothing
                if abs(self.position) == -1: ## make useless
                    if debug:
                        print("Going FLAT: ",index, self.current_rate_bid)
                    self.position = 0 

        # move to the next time step...
        self.current_index += 1

        # and get the rates...
        self.current_rate_bid_norm, self.current_rate_bid, self.current_rate_ask_norm, self.current_rate_ask = self.data_store.get_bid_ask(self.iday, self.current_index)

        value = 0.

        # LONG: CURRENT_BID - INITIAL_ASK
        if self.position > 0.1:
            value = self.position * (self.current_rate_bid - last_rate_bid)

        # SHORT: CURRENT_ASK - INITIAL_BID
        if self.position < -0.1:
            value = self.position * (self.current_rate_ask - last_rate_ask) 

        # get the reward
        reward += value
        self.daily_reward += reward
        # TODO test reward clipping?

        # integer 0-255 format 84 * 84
        # screen 84 * 84 = 7056
        # 250 * 4 = 1000
        # 1764 * 4= 1763; sinus 1763 + 6000 lang, sequence = 1764 als kopie, die ersten 4 bytes auf 0 setzen, dort die positionm reinschreiben und den einstandskurs.
        # sequence mit der laenge aus dem Data Store, dann die position und den init rate norm drauf schlagen in position 1763.
        # resizen auf 84 * 84.

        inputs = self.data_store.get_sequence(self.iday, self.current_index).copy()
  
        inputs[0,0] = self.position

        screen = np.resize(inputs, (84,8))

        return reward, terminal, screen # Screen is 84 x 84 i
