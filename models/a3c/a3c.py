# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

import threading
import signal
import math
import os
import time

from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE
from constants import PARALLEL_SIZE
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import USE_LSTM

device = "/gpu:0" if USE_GPU else "/cpu:0"
print("Conf: USING Device ", device)

def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                    INITIAL_ALPHA_HIGH,
                                    INITIAL_ALPHA_LOG_RATE)

global_t = 0

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
print("Conf: PARALLEL_SIZE: ", PARALLEL_SIZE)

training_threads = []
for i in range(PARALLEL_SIZE):
  training_thread = A3CTrainingThread(i, global_network, initial_learning_rate,
                                      learning_rate_input,
                                      grad_applier, MAX_TIME_STEP,
                                      device = device)
  training_threads.append(training_thread)

# prepare session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True))

init = tf.initialize_all_variables()
sess.run(init)

# summary for tensorboard
score_input = tf.placeholder(tf.int32)
tf.scalar_summary("score", score_input)

summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter(LOG_FILE, sess.graph_def)

# init or load checkpoint with saver
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print ("Conf: checkpoint loaded:", checkpoint.model_checkpoint_path)
  tokens = checkpoint.model_checkpoint_path.split("-")
  # set global step
  global_t = int(tokens[1])
  print (">>> global step set: ", global_t)
else:
  print ("Conf: Could not find old checkpoint")


# Ctrl+C handling

stop_requested = False
    
def signal_handler(signal, frame):
  global stop_requested
  print('Conf: You pressed Ctrl+C!')
  stop_requested = True
  
signal.signal(signal.SIGINT, signal_handler)


def train_function(parallel_index):
  global global_t

  while not stop_requested and global_t <= MAX_TIME_STEP:
    diff_global_t = training_threads[parallel_index].process(sess, global_t, summary_writer,
                                            summary_op, score_input)
    global_t += diff_global_t


# Start threads
train_threads = []
for i in range(PARALLEL_SIZE):
  train_threads.append(threading.Thread(target=train_function, args=(i,)))

start_time = time.time()
for t in train_threads:
  t.start()

print('Press Ctrl+C to stop')
signal.pause()

# Wait for threads to end
for t in train_threads:
  t.join()
end_time = time.time()

print('Conf: Now saving data. Please wait. Steps:', global_t)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)

print("Total Time: ", end_time-start_time, ", per Timestep: ", (end_time-start_time)/global_t)

