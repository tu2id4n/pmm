import tensorflow as tf
import numpy as np
from stable_baselines.common.policies import BasePolicy
from stable_baselines.common.input import observation_input
from stable_baselines.a2c.utils import ortho_init
from stable_baselines.common import tf_util
from _common import _constants


def img_cnn(scaled_images, name='img', **kwargs):
    activ = tf.nn.relu
    print("scaled_images", scaled_images)
    layer_1 = activ(conv(scaled_images, name + 'c1', n_filters=16, filter_size=8,
                         stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs))
    print("layer1", layer_1)
    layer_2 = activ(conv(layer_1, name + 'c2', n_filters=32, filter_size=4,
                         stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs))
    print("layer2", layer_2)
    layer_3 = activ(conv(layer_2, name + 'c3', n_filters=64, filter_size=3,
                         stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs))
    print("layer3", layer_3)
    layer_3 = conv_to_fc(layer_3)
    print("fc", layer_3)
    return activ(linear(layer_3, name + 'fc', n_hidden=512, init_scale=np.sqrt(2)))


def simple_fc(scalars, name='sca', n_dim=256):
    activ = tf.nn.relu
    layer_1 = activ(
        linear(scalars, name + '1', n_hidden=n_dim, init_scale=np.sqrt(2)))
    layer_2 = activ(
        linear(layer_1, name + '2', n_hidden=n_dim, init_scale=np.sqrt(2)))
    return activ(linear(layer_2, name + '3', n_hidden=n_dim, init_scale=np.sqrt(2)))


class DFPPolicy(BasePolicy):
    def __init__(self, sess, ob_space, sc_space, me_space, g_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, scale=False, pgn_params=None,
                 obs_phs=None, sca_phs=None, mea_phs=None, goal_phs=None, future_size=6):
        super(DFPPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=scale,
                                        obs_phs=obs_phs)
        with tf.variable_scope("input_fc", reuse=False):
            # if gm_phs is None:
            #     self._gm_ph, self._processed_gm = observation_input(
            #         gm_space, n_batch, scale=scale)
            # else:
            #     self._gm_ph, self._processed_gm = gm_phs

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
            with tf.variable_scope('img_cnn', reuse=reuse):
                # CNN提取棋盘特征
                extracted_img = img_cnn(self.processed_obs)
                extracted_img = tf.layers.flatten(extracted_img)

            with tf.variable_scope('sca_fc', reuse=reuse):
                # 标量特征
                # extracted_sca = simple_fc(self.processed_sca)
                extracted_sca = tf.layers.flatten(self.processed_sca)
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

            activ = tf.nn.relu

            with tf.variable_scope('exp_fc', reuse=reuse):
                # expectation_stream
                expectation_stream_prev = activ(linear(
                    extracted_input, 'exp_prev', n_hidden=512, init_scale=np.sqrt(2)))
                expectation_stream = activ(linear(
                    expectation_stream_prev, 'exp', n_hidden=self.future_size, init_scale=np.sqrt(2)))

            if _constants.pgn:
                print()
                print("PGN and DFP...")
                print()
                if pgn_params:
                    print("PGN Loading...")

                    len_params = len(pgn_params)
                    prev = [[None] * (_constants.n_actions - 1)] * len_params
                    prev_stream = [[None] * (_constants.n_actions - 1)] * len_params
                    for c in range(len_params):  # c代表第几列网络
                        with tf.variable_scope('prev_fc' + str(c), reuse=reuse):
                            for r in range(_constants.n_actions - 1):  # r代表动作 没有最后一个动作
                                scope1 = 'action_fc/act_prev' + str(r)
                                scope2 = 'action_fc/act' + str(r)
                                prev[c][r] = activ(pgn_linear(extracted_input, scope1,
                                                              ww=pgn_params[c][scope1 + '/w'],
                                                              bb=pgn_params[c][scope1 + '/b']))
                                prev_stream[c][r] = activ(pgn_linear(prev[c][r], scope2,
                                                                     ww=pgn_params[c][scope2 + '/w'],
                                                                     bb=pgn_params[c][scope2 + '/b']))

                    action_prev = [None] * _constants.n_actions
                    action_stream = [None] * _constants.n_actions

                    with tf.variable_scope('action_fc', reuse=reuse):
                        for i in range(_constants.n_actions - 1):  # 第 i 个动作
                            action_prev[i] = linear(
                                extracted_input, 'act_prev' + str(i), n_hidden=512, init_scale=np.sqrt(2))
                            for c in range(len_params):
                                action_prev[i] = tf.add(action_prev[i], prev[c][i])
                            action_prev[i] = activ(tf.divide(action_prev[i], len_params + 1))

                            action_stream[i] = linear(
                                action_prev[i], 'act' + str(i), n_hidden=self.future_size, init_scale=np.sqrt(2))
                            for c in range(len_params):
                                action_stream[i] = tf.add(action_stream[i], prev_stream[c][i])

                            action_stream[i] = activ(tf.divide(action_stream[i], len_params + 1))

                        # 最后一个放置炸弹单独处理
                        action_prev[121] = activ(linear(
                            extracted_input, 'act_prev' + str(121), n_hidden=512, init_scale=np.sqrt(2)))
                        action_stream[121] = activ(linear(
                            action_prev[121], 'act' + str(121), n_hidden=self.future_size, init_scale=np.sqrt(2)))
                else:
                    print("DFP Loading...")

                    action_prev = [None] * _constants.n_actions
                    action_stream = [None] * _constants.n_actions

                    with tf.variable_scope('action_fc', reuse=reuse):
                        for i in range(_constants.n_actions):
                            action_prev[i] = activ(linear(
                                extracted_input, 'act_prev' + str(i), n_hidden=512, init_scale=np.sqrt(2)))
                            action_stream[i] = activ(linear(
                                action_prev[i], 'act' + str(i), n_hidden=self.future_size, init_scale=np.sqrt(2)))

            else:
                print()
                print("Pure DFP...")
                print()

                action_prev = [None] * _constants.n_actions
                action_stream = [None] * _constants.n_actions

                with tf.variable_scope('action_fc', reuse=reuse):
                    for i in range(_constants.n_actions):
                        action_prev[i] = activ(linear(
                            extracted_input, 'act_prev' + str(i), n_hidden=512, init_scale=np.sqrt(2)))
                        action_stream[i] = activ(linear(
                            action_prev[i], 'act' + str(i), n_hidden=self.future_size, init_scale=np.sqrt(2)))

            n_actions = len(action_stream)

            # 求 sum
            action_sum = action_stream[0]
            for i in range(1, n_actions):
                action_sum = tf.add(action_sum, action_stream[i])
            # 求 mean
            action_mean = tf.divide(action_sum, n_actions)
            #
            for i in range(n_actions):
                action_stream[i] = tf.subtract(action_stream[i], action_mean)
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
                                {self.obs_ph: imgs, self.sca_ph: scas, self.mea_ph: meas,
                                 self.goal_ph: goals})
        return futures

    def get_futures(self, imgs, scas, meas, goals):
        futures = self.sess.run(self.futures,
                                {self.obs_ph: imgs, self.sca_ph: scas, self.mea_ph: meas,
                                 self.goal_ph: goals})
        return futures

    def mse_loss(self, targets):
        error = self.futures - targets
        mse_error = tf_util.huber_loss(error)

        return mse_error, self.futures

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


def conv(input_tensor, scope, *, n_filters, filter_size, stride,
         pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    """
    Creates a 2d convolutional layer for TensorFlow

    :param input_tensor: (TensorFlow Tensor) The input tensor for the convolution
    :param scope: (str) The TensorFlow variable scope
    :param n_filters: (int) The number of filters
    :param filter_size:  (Union[int, [int], tuple<int, int>]) The filter size for the squared kernel matrix,
    or the height and width of kernel filter if the input is a list or tuple
    :param stride: (int) The stride of the convolution
    :param pad: (str) The padding type ('VALID' or 'SAME')
    :param init_scale: (int) The initialization scale
    :param data_format: (str) The data format for the convolution weights
    :param one_dim_bias: (bool) If the bias should be one dimentional or not
    :return: (TensorFlow Tensor) 2d convolutional layer
    """
    if isinstance(filter_size, list) or isinstance(filter_size, tuple):
        assert len(filter_size) == 2, \
            "Filter size must have 2 elements (height, width), {} were given".format(len(filter_size))
        filter_height = filter_size[0]
        filter_width = filter_size[1]
    else:
        filter_height = filter_size
        filter_width = filter_size
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, n_filters]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, n_filters, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [n_filters] if one_dim_bias else [1, n_filters, 1, 1]
    n_input = input_tensor.get_shape()[channel_ax].value
    wshape = [filter_height, filter_width, n_input, n_filters]
    with tf.variable_scope(scope):
        weight = tf.get_variable("w", wshape, initializer=_constants.conv_init, dtype=tf.float32)
        bias = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        if not one_dim_bias and data_format == 'NHWC':
            bias = tf.reshape(bias, bshape)
        return bias + tf.nn.conv2d(input_tensor, weight, strides=strides, padding=pad, data_format=data_format)


def linear(input_tensor, scope, n_hidden, *, init_scale=1.0, init_bias=0.0):
    """
    Creates a fully connected layer for TensorFlow

    :param input_tensor: (TensorFlow Tensor) The input tensor for the fully connected layer
    :param scope: (str) The TensorFlow variable scope
    :param n_hidden: (int) The number of hidden neurons
    :param init_scale: (int) The initialization scale
    :param init_bias: (int) The initialization offset bias
    :return: (TensorFlow Tensor) fully connected layer
    """
    with tf.variable_scope(scope):
        n_input = input_tensor.get_shape()[1].value
        weight = tf.get_variable("w", [n_input, n_hidden], initializer=_constants.linear_init, dtype=tf.float32)
        bias = tf.get_variable("b", [n_hidden], initializer=tf.constant_initializer(init_bias), dtype=tf.float32)
        return tf.matmul(input_tensor, weight) + bias


def pgn_linear(input_tensor, scope, *, ww=None, bb=None):
    with tf.variable_scope(scope):
        weight_fix = tf.convert_to_tensor(ww, dtype=tf.float32)
        bias_fix = tf.convert_to_tensor(bb, dtype=tf.float32)
        weight = tf.get_variable("w", initializer=weight_fix, trainable=False)
        bias = tf.get_variable("b", initializer=bias_fix, trainable=False)

        return tf.matmul(input_tensor, weight) + bias


def conv_to_fc(input_tensor):
    """
    Reshapes a Tensor from a convolutional network to a Tensor for a fully connected network

    :param input_tensor: (TensorFlow Tensor) The convolutional input tensor
    :return: (TensorFlow Tensor) The fully connected output tensor
    """
    n_hidden = np.prod([v.value for v in input_tensor.get_shape()[1:]])
    input_tensor = tf.reshape(input_tensor, [-1, n_hidden])
    return input_tensor
