from stable_baselines.a2c.utils import total_episode_reward_logger
import time
import sys
import multiprocessing
from collections import deque

import gym
import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import BaseRLModel, tf_util, SetVerbosity, TensorboardWriter

import copy
from tqdm import tqdm
import random
from _policies import DFPPolicy
from _common import featurize, model_utils
from .replay_buffer import ReplayBuffer
from stable_baselines.common.policies import ActorCriticPolicy


class DFP(BaseRLModel):
    def __init__(self, policy=DFPPolicy, env=None, gamma=0.99, learning_rate=5e-4, buffer_size=50000,
                 learning_starts=1000, time_spans=[1, 3, 6, 9, 12], hindsight=True,
                 exploration_fraction=0.1, exploration_final_eps=0.02, batch_size=32, n_steps=128, nminibatches=4,
                 verbose=0, tensorboard_log=None, full_tensorboard_log=False, _init_setup_model=True,
                 policy_kwargs=None):

        super(DFP, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                  policy_kwargs=policy_kwargs, policy_base=ActorCriticPolicy)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.exploration_fraction = exploration_fraction
        self.exploration_final_eps = exploration_final_eps
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.nminibatches = nminibatches  #
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        self.graph = None
        self.act_model = None
        self.train_model = None
        self.params = None
        self.loss = None

        self.replay_buffer = None
        self.hindsight = hindsight
        self.exploration = None
        self.summay = None
        self.episode_reward = None
        self.wins = None
        self.eps = None

        self.img_space = featurize.get_img_space()
        self.scas_space = featurize.get_scas_space()
        self.meas_space = featurize.get_meas_space()
        self.goal_space = featurize.get_goal_space()
        self.action_space = featurize.get_action_space()
        self.goalmap_space = featurize.get_goalmap_space()
        self.n_actions = self.action_space.n
        self.time_spans = time_spans
        self.future_len = len(self.time_spans)
        self.meas_size = self.meas_space.shape[0]
        self.future_size = self.future_len * self.meas_size

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        with SetVerbosity(self.verbose):
            # n_cpu = multiprocessing.cpu_count()
            # if sys.platform == 'darwin':
            #     n_cpu //= 2

            self.graph = tf.Graph()  # 创建图
            with self.graph.as_default():
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                n_batch_step = None
                n_batch_train = None

                act_model = self.policy(self.sess, self.img_space, self.scas_space, self.meas_space, self.goal_space,
                                        self.action_space, self.goalmap_space,
                                        self.n_envs, 1, n_batch_step, reuse=False, future_size=self.future_size,
                                        **self.policy_kwargs)

                with tf.variable_scope('train_model', reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.img_space, self.scas_space, self.meas_space,
                                              self.goal_space, self.action_space, self.goalmap_space,
                                              self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                              reuse=False, future_size=self.future_size,
                                              **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    self.targets_ph = tf.placeholder(tf.float32, [self.n_actions, None, self.future_size],
                                                     name="targets")
                    mse_error = train_model.mse_loss(self.targets_ph)
                    self.mse = mse_error
                    self.loss = tf.reduce_mean(mse_error)
                    tf.summary.scalar('loss', self.loss)

                    with tf.variable_scope('model'):
                        self.params = tf.trainable_variables()

                    grads = tf.gradients(self.loss, self.params)
                    grads = list(zip(grads, self.params))
                    # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                    self._train = optimizer.apply_gradients(grads)

                tf.global_variables_initializer().run(session=self.sess)
                self.act_model = act_model
                self.train_model = train_model
                self.summay = tf.summary.merge_all()

                # print('---------')
                # print([i.name for i in tf.global_variables()])
                # print('---------')
                # print([i.name for i in tf.local_variables()])

            #  输出计算图
            # print([n.name for n in self.graph.as_graph_def().node])
            # writer = tf.summary.FileWriter("./log/graphs", self.sess.graph)
            # writer.flush()
            # writer.close()

    def learn(self, total_timesteps, callback=None, tb_log_name="DFP", reset_num_timesteps=True, save_interval=10000,
              save_path=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        save_interval_st = save_interval

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn()  # 检验是否有环境
            self.replay_buffer = ReplayBuffer(self.buffer_size, self.time_spans, self.batch_size)

            self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                              initial_p=1,
                                              final_p=self.exploration_final_eps)

            obs = self.env.reset()  # (imgs, scas, meas, goal, gms)  ==> # [(11, 11, 10), (6, ), (4, ), (4, ), (11, 11, 3)]
            reset = True  #
            self.episode_reward = np.zeros((1,))
            self.wins = np.zeros((1,))

            for _ in tqdm(range(total_timesteps)):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                update_eps = self.exploration.value(self.num_timesteps)

                with self.sess.as_default():
                    futures = self.act_model.step(obs)  # (n_act, n_batch, future_size)
                futures = self.convert_futures(futures)
                action = self.make_action(obs[0][3], futures[0], update_eps=update_eps)
                action = np.array([action])

                new_obs, rew, done, terminal_obs, win = self.env.step([(action, 0)])
                self.replay_buffer.add(obs[0], action[0], rew[0], done[0], terminal_obs[0], win[0], hindsight=self.hindsight)
                obs = new_obs
                if writer is not None:
                    summary_eps = tf.Summary(value=[tf.Summary.Value(tag='update_eps', simple_value=update_eps)])
                    writer.add_summary(summary_eps, self.num_timesteps)

                    self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                                      rew.reshape((self.n_envs, 1)),
                                                                      done.reshape((self.n_envs, 1)),
                                                                      writer, self.num_timesteps)
                    self.wins = model_utils.total_rate_logger(self.wins,
                                                              win.reshape((self.n_envs, 1)),
                                                              done.reshape((self.n_envs, 1)),
                                                              writer, self.num_timesteps,
                                                              name='win_rate')

                can_sample = self.replay_buffer.can_sample()

                if can_sample and self.num_timesteps > self.learning_starts:
                    # print("Sampling ...")
                    imgs, scas, meas, goals, gms, actions, _futures = self.replay_buffer.sample()
                    if writer is not None:
                        # print("Training ...")
                        target_futures = self.act_model.get_futures(imgs, scas, meas, goals, gms)
                        targets = self.get_targets(actions, _futures, target_futures)
                        # np_mse = []
                        # for i in range(self.action_space.n):
                        #     mse = target_futures[i] - targets[i]
                        #     print('mse', 0.5 * np.square(mse))
                        #     np_mse.append(0.5 * np.square(mse))

                        # print("np.loss", np.mean(np.array(np_mse)))

                        td_map = {self.train_model.obs_ph: imgs, self.train_model.sca_ph: scas,
                                  self.train_model.mea_ph: meas, self.train_model.goal_ph: goals,
                                  self.targets_ph: targets, self.train_model.gm_ph: gms,
                                  }

                        summary, loss, _, mse = self.sess.run([self.summay, self.loss, self._train, self.mse], td_map)
                    writer.add_summary(summary, self.num_timesteps)
                    # print(self.sess.run(self.params[9]))

                if self.num_timesteps >= save_interval_st:
                    save_interval_st += save_interval
                    s_path = save_path + '_' + str(int(self.num_timesteps / save_interval)) + 'M.zip'
                    self.save(save_path=s_path)

                self.num_timesteps += 1

    def convert_futures(self, futures):
        return futures.swapaxes(0, 1)

    def make_action(self, goal, futures, update_eps=0):
        if random.random() < update_eps:
            return random.randint(1, 4)  # WASD

        actions = []
        goals = np.tile(goal, self.future_len)
        for f in futures:
            actions.append(goals.dot(f))
        
        return np.argmax(np.array(actions))

    def get_targets(self, actions, futures, _target_futures):
        target_futures = copy.deepcopy(_target_futures)
        for i in range(self.batch_size):
            target_futures[actions[i]][i] = futures[i]

        # 将小于0的置为0
        # target_futures = np.where(target_futures > 0, target_futures, 0)

        return target_futures

    def predict(self, obs):
        obs = np.array(obs).reshape(1, -1)
        futures = self.act_model.step(obs)
        futures = self.convert_futures(futures)
        # print(futures)
        action = self.make_action(obs[0][3], futures[0], 0)
        return action

    def save(self, save_path, cloudpickle=False):
        print()
        print("Saving...")
        print(save_path)
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "learning_rate": self.learning_rate,
            "learning_starts": self.learning_starts,
            "buffer_size": self.buffer_size,
            "exploration_final_eps": self.exploration_final_eps,
            "exploration_fraction": self.exploration_fraction,
            "batch_size": self.batch_size,
            "nminibatches": self.nminibatches,
            "verbose": self.verbose,
            "policy": self.policy,
            "img_space": self.img_space,
            "scas_space": self.scas_space,
            "meas_space": self.meas_space,
            "goal_space": self.goal_space,
            "action_space": self.action_space,
            "goalmap_space": self.goalmap_space,
            "n_actions": self.n_actions,
            "time_spans": self.time_spans,
            "future_len": self.future_len,
            "meas_size": self.meas_size,
            "future_size": self.future_size,
            "n_envs": self.n_envs,
            "hindsight:": self.hindsight,
            "policy_kwargs": self.policy_kwargs,
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)

    @classmethod
    def load(cls, load_path, env=None, tensorboard_log=None, custom_objects=None, **kwargs):
        print("Loading...")
        print(load_path)
        data, params = cls._load_from_file(load_path, custom_objects=custom_objects)
        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(policy=data["policy"], env=None, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        if env:
            # model.set_env(env)
            model.env = env
        model.tensorboard_log = tensorboard_log
        model.setup_model()
        model.load_parameters(params)

        return model

    def _get_pretrain_placeholders(self):
        pass

    def action_probability(self):
        pass

    def get_parameter_list(self):
        return self.params


class Schedule(object):
    def value(self, step):
        """
        Value of the schedule for a given timestep

        :param step: (int) the timestep
        :return: (float) the output value for the given timestep
        """
        raise NotImplementedError


class LinearSchedule(Schedule):
    """
    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.

    :param schedule_timesteps: (int) Number of timesteps for which to linearly anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    """

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, step):
        fraction = min(float(step) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
