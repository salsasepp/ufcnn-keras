# -*- coding: utf-8 -*-
import sys
import numpy as np

from constants import SEQUENCE_LENGTH
from constants import FEATURES_LIST
from constants import TRAINING_DAYS
from constants import TESTING_DAYS

from DataStore import DataStore
from Trading import Trading


class GameState(object):

    def __init__(self, rand_seed, display=False, testing=False, show_trades=None):
        np.random.seed(rand_seed)

        # Load the data
        training_store = DataStore(
            training_days=TRAINING_DAYS, features_list=FEATURES_LIST, sequence_length=SEQUENCE_LENGTH)

        if testing:
            print("Set up for testing")
            testing_store = DataStore(
                training_days=TRAINING_DAYS, testing_days=TESTING_DAYS, features_list=FEATURES_LIST,
                                     sequence_length=SEQUENCE_LENGTH, mean=training_store.mean, std=training_store.std)
            self.environment = Trading(testing_store, sequence_length=SEQUENCE_LENGTH, testing=testing, show_trades=show_trades)
        else:
            self.environment = Trading(training_store, sequence_length=SEQUENCE_LENGTH, testing=testing, show_trades=show_trades)
        self.old_x = 0.

        self.reset()

    def _process_frame(self, action):
        # reward = self.environment.act(action)
        # terminal = self.environment.game_over()

        reward, terminal, _screen = self.environment.get_reward(action)

        # TODO: What is this?
        x_t = _screen
        x_t = x_t.astype(np.float32)

        x1 = x_t[1, 0]
        x2 = x_t[2, 0]

        # print(x_t)

        # if (x1 != self.old_x):
        #    print(" X error", x1, x2)
        self.old_x = x2

        return reward, terminal, x_t

    def _setup_display(self):
        print("setup_display() is not implemented...")

    def reset(self):
        self.environment.reset()

        # TODO: randomize initial state?

        self.reward = 0
        self.terminal = False

        _, _, x_t = self._process_frame(2)  # Action GO_FLAT
        self.s_t = self.rebase(x_t)

    def process(self, action):
        self.reward, self.terminal, x_t1 = self._process_frame(action)
        self.s_t1 = self.rebase(x_t1)

    def rebase(self, x):
        return x if len(x.shape) == 3 else np.expand_dims(x, axis=2)

    def update(self):
        self.s_t = self.s_t1

