import random
import numpy as np
from _common import _constants
import copy

_img = 0
_sca = 1
_mea = 2
_goal = 3
_done = 7
_tmea = 11

_idx_map = 2
_goal_map = 3


class ReplayBuffer(object):
    def __init__(self, buffer_size, time_spans, batch_size):
        self.storage = []
        self.maxsize = buffer_size
        self.time_spans = time_spans
        self.future_size = len(time_spans)
        self.batch_size = batch_size
        self.her_size = _constants.her_size
        self.K = _constants.her_K
        self.her = 0
        self.her_storage = []
        self.hindsight = _constants.hindsight
        if self.hindsight:
            print()
            print("Using HindSight to collect trajectories...")

    def can_sample(self):
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self.storage) >= 2 * self.batch_size

    def add(self, obs, actions, rews, dones, terminal_obs, wins, states):
        imgs, scas, meas, goals = obs
        t_imgs, t_scas, t_meas, t_goals = terminal_obs
        data = (imgs, scas, meas, goals, actions, rews, dones,
                wins, t_imgs, t_scas, t_meas, t_goals, states)

        self.storage.append(data)
        while len(self.storage) > self.maxsize:
            self.storage.pop(0)

        # self.her += 1
        # if self.hindsight and not dones and self.her > self.her_size:
        #     self.her = 0
        #     st = len(self.storage) - 1 - self.her_size
        #     f_data = []
        #
        #     while len(self.her_storage) > self.maxsize:
        #         self.her_storage.pop(0)  # 弹出buffer第一位

    def sample(self):
        return self.seq_sample()

    def seq_sample(self):
        _len = len(self.storage) - 1 - self.time_spans[-1]
        her_len = len(self.her_storage) - 1 - self.time_spans[-1]
        imgs, scas, meas, goals, actions, futures = [], [], [], [], [], []

        storage = self.storage
        for i in range(self.batch_size):
            her_flag = False
            if self.hindsight and i >= self.batch_size / 2:
                _len = her_len
                storage = self.her_storage
                her_flag = True

            idx = random.randint(0, _len)
            data = storage[idx]
            while data[_done]:
                idx = random.randint(0, _len)
                data = storage[idx]

            img, sca, mea, goal, action, rew, done, win, t_img, t_sca, t_mea, t_goal, state = data
            imgs.append(img)
            scas.append(sca)
            meas.append(mea)
            goals.append(goal)

            actions.append(action)
            future = self.compute_future(idx=idx, storage=storage, her_flag=her_flag)
            futures.append(future)

        return np.array(imgs), np.array(scas), np.array(meas), np.array(goals), \
               np.array(actions), np.array(futures)

    def compute_future(self, idx, storage, her_flag):
        future = []
        cur_mea = storage[idx][_mea]
        j = idx
        terminal = False
        while j - idx <= self.time_spans[-1]:
            _, _, j_mea, _, _, _, j_done, _, \
            _, _, _, _, _ = storage[j]
            if j_done and not terminal:  # 代表结束
                terminal = True
                terminal_mea = j_mea
            if (j - idx) in self.time_spans:  # 如果在 timespans 内
                if terminal:
                    diff_mea = terminal_mea - cur_mea
                else:
                    diff_mea = j_mea - cur_mea
                future.extend(diff_mea)

                # 如果出错了
                if diff_mea[0] < 0:
                    print('her_flag', her_flag)
                    print('idx', idx)
                    print('cur_mea', storage[idx][_mea])
                    k = idx
                    while k - idx <= self.time_spans[-1]:
                        print('k-idx', k - idx, storage[k][_mea], storage[k][_done])
                        k += 1

            j += 1
        return np.array(future)
