# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from a3c_util import choose_action
from accum_trainer import AccumTrainer
from game_state import GameState
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork

from constants import ACTION_SIZE
from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import USE_LSTM


class A3CTrainingThread(object):

    def __init__(self,
                 thread_index,
                 global_network,
                 initial_learning_rate,
                 learning_rate_input,
                 grad_applier,
                 max_global_time_step,
                 device):

        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step

        if USE_LSTM:
            self.local_network = GameACLSTMNetwork(ACTION_SIZE, thread_index, device)
        else:
            self.local_network = GameACFFNetwork(ACTION_SIZE, device)

        self.local_network.prepare_loss(ENTROPY_BETA)

        # TODO: don't need accum trainer anymore with batch
        self.trainer = AccumTrainer(device)
        self.trainer.prepare_minimize(self.local_network.total_loss,
                                      self.local_network.get_vars())

        self.accum_gradients = self.trainer.accumulate_gradients()
        self.reset_gradients = self.trainer.reset_gradients()

        self.apply_gradients = grad_applier.apply_gradients(
            global_network.get_vars(),
          self.trainer.get_accum_grad_list())

        self.sync = self.local_network.sync_from(global_network)

        self.game_state = GameState(113 * thread_index)

        self.local_t = 0

        self.initial_learning_rate = initial_learning_rate

        self.episode_reward = 0

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * \
            (self.max_global_time_step - global_time_step) / \
             self.max_global_time_step
        assert learning_rate > 0, 'Learning rate {} is not >0'.format(
            learning_rate)
        return learning_rate

    def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
        summary_str = sess.run(summary_op, feed_dict={
                               score_input: score
                               })
        summary_writer.add_summary(summary_str, global_t)

    def process(self, sess, global_t, summary_writer, summary_op, score_input):
        states = []
        actions = []
        rewards = []
        values = []

        # reset accumulated gradients
        sess.run(self.reset_gradients)

        # copy weights from shared to local
        sess.run(self.sync)

        if USE_LSTM:
            start_lstm_state = self.local_network.lstm_state_out

        # t_max times loop
        start_local_t = self.local_t
        terminal_end = False
        for i in range(LOCAL_T_MAX):
            pi_, value_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t)
            action = choose_action(pi_)

            states.append(self.game_state.s_t)
            actions.append(action)
            values.append(value_)

            # Debug output for progress
            if (self.thread_index == 0) and (self.local_t % 100) == 0:
                print(('local_t = {:10}  pi = ' + '{:7.5f} ' * len(pi_) + ' V = {:8.4f} (thread {})').format(self.local_t,
                                                                                                             *pi_, value_, self.thread_index))

            # process game
            self.game_state.process(action)

            # receive game result
            reward = self.game_state.reward
            terminal = self.game_state.terminal

            self.episode_reward += reward

            # clip reward
            # TODO: Does this make sense?
            rewards.append(np.clip(reward, -1, 1))

            self.local_t += 1

            # s_t1 -> s_t
            self.game_state.update()

            if terminal:
                terminal_end = True
                print ("score=", self.episode_reward)

                self._record_score(
                    sess, summary_writer, summary_op, score_input,
                                   self.episode_reward, global_t)

                self.episode_reward = 0
                self.game_state.reset()
                if USE_LSTM:
                    self.local_network.reset_state()
                break

        R = 0.0 if terminal_end else self.local_network.run_value(sess, self.game_state.s_t)

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_si = []
        batch_a = []
        batch_td = []
        batch_R = []

        # compute and accmulate gradients
        for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
            R = ri + GAMMA * R
            td = R - Vi
            a = np.zeros([ACTION_SIZE])
            a[ai] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        if USE_LSTM:
            batch_si.reverse()
            batch_a.reverse()
            batch_td.reverse()
            batch_R.reverse()

            sess.run(self.accum_gradients,
                     feed_dict={
                     self.local_network.s: batch_si,
                     self.local_network.a: batch_a,
                     self.local_network.td: batch_td,
                     self.local_network.r: batch_R,
                     self.local_network.initial_lstm_state: start_lstm_state,
                     self.local_network.step_size: [len(batch_a)]})
        else:
            sess.run(self.accum_gradients,
                     feed_dict={
                     self.local_network.s: batch_si,
                     self.local_network.a: batch_a,
                     self.local_network.td: batch_td,
                     self.local_network.r: batch_R})

        cur_learning_rate = self._anneal_learning_rate(global_t)

        sess.run(self.apply_gradients,
                 feed_dict={self.learning_rate_input: cur_learning_rate})

        if (self.thread_index == 0) and (self.local_t % 100) == 0:
            print ("TIMESTEP", self.local_t)

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t
