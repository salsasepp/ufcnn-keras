# -*- coding: utf-8 -*-
import sys
import numpy as np
#import cv2
#from ale_python_interface import ALEInterface


from constants import ACTION_SIZE
from constants import SEQUENCE_LENGTH
from constants import FEATURES_LIST
from constants import STORE_PATH
from constants import TRAINING_DAYS
from constants import TESTING_DAYS

from DataStore import DataStore 
from Trading import Trading 


class GameState(object):
  def __init__(self, rand_seed, display=False, no_op_max=27, testing=False, show_trades=None):
    #self.ale = ALEInterface()

    np.random.seed(rand_seed)
    self._no_op_max = no_op_max

    self.sequence_length = SEQUENCE_LENGTH
    features_list = FEATURES_LIST
    self.features_length = len(FEATURES_LIST)
    path = STORE_PATH
    training_days = TRAINING_DAYS
    testing_days = TESTING_DAYS

    # Load the data
    training_store = DataStore(training_days=training_days, features_list=features_list, sequence_length=self.sequence_length)

    if testing:
        print("Set up for testing")
        testing_store = DataStore(training_days=training_days, testing_days=testing_days, features_list=features_list, 
            sequence_length=self.sequence_length, mean=training_store.mean, std=training_store.std)
        self.environment = Trading(data_store=testing_store, sequence_length=self.sequence_length, features_length=self.features_length, testing=testing)
    else:
        self.environment = Trading(data_store=training_store, sequence_length=self.sequence_length, features_length=self.features_length, testing=testing, show_trades=show_trades)
    self.old_x = 0.

    # collect minimal action set
    #self.real_actions = self.ale.getMinimalActionSet()
    self.real_actions = [0, 1, 2]

    # height=210, width=160

    self.reset()

  def _process_frame(self, action, reshape):
    #reward = self.environment.act(action)
    #terminal = self.environment.game_over()

    reward, terminal, _screen = self.environment.get_reward(action) # Screen is 84 x 84
    x_t = _screen

    # screen shape is (210, 160, 1)
    #self.ale.getScreenGrayscale(self._screen)
    
    # reshape it into (210, 160)
    #reshaped_screen = np.reshape(self._screen, (210, 160))
    
    # resize to height=110, width=84
    #resized_screen = cv2.resize(reshaped_screen, (84, 110))
    
    #x_t = resized_screen[18:102,:]

    x_t = x_t.astype(np.float32)

    x1 = x_t[1,0]
    x2 = x_t[2,0]

    #print(x_t)
   

    if reshape:
        #x_t = np.reshape(x_t, (x_t.shape[0],1 , x_t.shape[1]))
        pass

  
    #if (x1 != self.old_x):
    #    print(" X error", x1, x2)
    self.old_x = x2

    #x_t *= (1.0/255.0)
    return reward, terminal, x_t
    
    
  def _setup_display(self):
    print("setup_display() is not implemented...")

  def reset(self):
    #self.ale.reset_game()
    self.environment.reset()
    
    # randomize initial state
    if self._no_op_max > 0:
      no_op = np.random.randint(0, self._no_op_max + 1)
      for _ in range(no_op):
         _, __, ___ = self.environment.get_reward(2) # Screen is 84 x 84


    _, _, x_t = self._process_frame(2, False) # Action GO_FLAT
    
    self.reward = 0
    self.terminal = False
    #self.s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
    self.s_t = self.rebase(x_t)
    
  def process(self, action):
    # convert original 18 action index to minimal action set index
    real_action = self.real_actions[action] # WHAT DOES THIS DO
    
    r, t, x_t1 = self._process_frame(real_action, True)

    self.reward = r
    self.terminal = t

    # stacked ...
    #self.s_t1 = np.append(self.s_t[:,:,1:], x_t1, axis = 2)    
    self.s_t1 = self.rebase(x_t1)
    # look at this here...

  def rebase(self, x):
    if len(x.shape) == 3:
       return x
    else:
       return np.expand_dims(x, axis=2)
     
  def update(self):
    self.s_t = self.s_t1
