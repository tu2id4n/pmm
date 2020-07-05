import tensorflow as tf
import numpy as np
from gym.spaces import Discrete, Box
from stable_baselines.common.policies import BasePolicy, nature_cnn, register_policy
from stable_baselines.common.input import observation_input
from abc import ABC, abstractmethod
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.common import tf_util

def simple_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8,
                         stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4,
                         stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3,
                         stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


def simple_fc(scalars, name='sca', **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(
        linear(scalars, name+'1', n_hidden=128, init_scale=np.sqrt(2)))
    layer_2 = activ(
        linear(scalars, name+'2', n_hidden=128, init_scale=np.sqrt(2)))
    return activ(linear(layer_2, name+'3', n_hidden=128, init_scale=np.sqrt(2)))


class DFPPolicy(BasePolicy):
    def __init__(self, sess, ob_space, sc_space, me_space, g_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False,
                 obs_phs=None, sca_phs=None, mea_phs=None, goal_phs=None, future_size=6):
        super(DFPPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=scale,
                                        obs_phs=obs_phs)
        with tf.variable_scope("input_fc", reuse=False):
            if sca_phs is None:
                self._sca_ph, self._processed_sca = observation_input(
                    sc_space, n_batch, scale=scale)
            else:
                self._sca_ph, self._processed_sca = sca_phs

            if mea_phs is None:
                self._mea_ph, self._processed_mea = observation_input(
                    me_space, n_batch, scale=scale)
            else:
                self._mea_ph, self._processed_mea = mea_phs

            if goal_phs is None:
                self._goal_ph, self._processed_goal = observation_input(
                    g_space, n_batch, scale=scale)
            else:
                self._goal_ph, self._processed_goal = goal_phs

            self._action_ph = None

        self.n_actions = ac_space.n
        self.future_size = future_size

        with tf.variable_scope('model', reuse=reuse):
            with tf.variable_scope('cnn', reuse=reuse):
                # CNN提取棋盘特征
                extracted_img = simple_cnn(self.processed_obs)
                extracted_img = tf.layers.flatten(extracted_img)
            with tf.variable_scope('sca_fc', reuse=reuse):
                # 标量特征
                extracted_sca = simple_fc(self.processed_sca)
                extracted_sca = tf.layers.flatten(extracted_sca)
            with tf.variable_scope('mea_fc', reuse=reuse):
                # 衡量值特征
                extracted_mea = simple_fc(self.processed_mea, name='mea')
                extracted_mea = tf.layers.flatten(extracted_mea)
            with tf.variable_scope('goal_fc', reuse=reuse):
                # goal特征
                extracted_goal = simple_fc(self.processed_goal, name='goal')
                extracted_goal = tf.layers.flatten(extracted_goal)

            with tf.variable_scope('concat', reuse=reuse):
                # 将所有特征拼接
                extracted_input = tf.concat(
                    [extracted_img, extracted_sca, extracted_mea, extracted_goal], axis=1, name='concat')

            with tf.variable_scope('exp_fc', reuse=reuse):
                # expectation_stream
                expectation_stream = tf.nn.tanh(linear(
                    extracted_input, 'exp', n_hidden=self.future_size, init_scale=np.sqrt(2)))

            # action_stream
            action_stream = [None] * self.n_actions
            for i in range(self.n_actions):
                with tf.variable_scope('action_fc' + str(i), reuse=reuse):
                    action_stream[i] = tf.nn.tanh(linear(
                        extracted_input, 'act' + str(i), n_hidden=self.future_size, init_scale=np.sqrt(2)))
                    action_stream[i] = tf.add(action_stream[i], expectation_stream)
            with tf.variable_scope('future', reuse=reuse):
                self._future_stream = tf.convert_to_tensor(action_stream)
                self._setup_init()

    def _setup_init(self):
        """
        Set up actions
        """
        with tf.variable_scope("output", reuse=True):
            self._futures = self.future_stream  # [n_act, n_batch, n_time_span]

    def step(self, obs):
        imgs, scas, meas, goals = zip(*obs)

        futures = self.sess.run(self.futures,
                                {self.obs_ph: imgs, self.sca_ph: scas, self.mea_ph: meas, self.goal_ph: goals})
        return futures

    def get_futures(self, imgs, scas, meas, goals):
        futures = self.sess.run(self.futures,
                                {self.obs_ph: imgs, self.sca_ph: scas, self.mea_ph: meas, self.goal_ph: goals})
        return futures

    def mse_loss(self, targets):
        error = self.futures - targets
        mse_error = tf_util.huber_loss(error)

        return mse_error

    @property
    def sca_ph(self):
        return self._sca_ph

    @property
    def mea_ph(self):
        return self._mea_ph
    
    @property
    def goal_ph(self):
        return self._goal_ph

    @property
    def processed_sca(self):
        return self._processed_sca

    @property
    def processed_mea(self):
        return self._processed_mea

    @property
    def processed_goal(self):
        return self._processed_goal

    @property
    def future_stream(self):
        return self._future_stream

    @property
    def futures(self):
        return self._futures

    def proba_step(self):
        pass



