# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random

from game_state import GameState
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE
from constants import PARALLEL_SIZE
from constants import CHECKPOINT_DIR
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import USE_LSTM

from constants import TESTING_DAYS

def choose_action(pi_values):
  values = []
  sum = 0.0
  for rate in pi_values:
    sum = sum + rate
    value = sum
    values.append(value)
    
  r = random.random() * sum
  for i in range(len(values)):
    if values[i] >= r:
      return i;
  #fail safe
  return len(values)-1

# use CPU for display tool
device = "/cpu:0"

if USE_LSTM:
  global_network = GameACLSTMNetwork(ACTION_SIZE, -1, device)
else:
  global_network = GameACFFNetwork(ACTION_SIZE, device)

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                              decay = RMSP_ALPHA,
                              momentum = 0.0,
                              epsilon = RMSP_EPSILON,
                              clip_norm = GRAD_NORM_CLIP,
                              device = device)

training_threads = []
for i in range(PARALLEL_SIZE):
  training_thread = A3CTrainingThread(i, global_network, 1.0,
                                      learning_rate_input,
                                      grad_applier,
                                      8000000,
                                      device = device)
  training_threads.append(training_thread)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print ("checkpoint loaded:", checkpoint.model_checkpoint_path)
else:
  print ("Could not find old checkpoint")

game_state = GameState(0, display=True, no_op_max=0, testing=True, show_trades=True)

testing_days = TESTING_DAYS
total_pnl = 0

for i in range(testing_days):
    print("Working on day ",i)
    terminal = False
    daily_pnl = 0

    #new
    if i > 0:
        game_state.environment.reset() 

    while not terminal:
        pi_values = global_network.run_policy(sess, game_state.s_t)

        action = choose_action(pi_values)
        game_state.process(action)

        reward = game_state.reward

        terminal = game_state.terminal

        game_state.update()

    game_state.environment.create_plot(game_state.environment.iday)
    daily_pnl = game_state.environment.daily_pnl
    total_pnl += daily_pnl
    game_state.environment.daily_pnl = 0

    print("Day ",i, ",Realized PnL: ", daily_pnl)
print("Total Realized PnL: ", total_pnl)


for i in range(testing_days):
    print("Potting day ",i)
    

