import random
import numpy as np
from _common import _constants

_img = 0
_sca = 1
_mea = 2
_goal = 3
_gm = 4
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

    def add(self, obs, actions, rews, dones, terminal_obs, wins):
        imgs, scas, meas, goals, gms = obs
        t_imgs, t_scas, t_meas, t_goals, t_gms = terminal_obs
        data = (imgs, scas, meas, goals, gms, actions, rews, dones,
                wins, t_imgs, t_scas, t_meas, t_goals, t_gms)
        self.storage.append(data)
        while len(self.storage) > self.maxsize:
            self.storage.pop(0)

        self.her += 1
        if self.hindsight and not dones and self.her > self.her_size:
            self.her = 0
            st = len(self.storage) - 1 - self.her_size
            f_data = []
            for t in range(self.her_size + 1):
                tmp_data = self.storage[st + t]
                if tmp_data[_done]:
                    f_data.append(tmp_data)
            f_data.append(self.storage[st + self.her_size])

            f_index = 0
            achive_count = 0
            for t in range(self.her_size):
                cur_data = self.storage[st + t]
                c_img, c_sca, c_mea, c_goal, c_gm, c_action, c_rew, c_done, c_win, \
                c_t_img, c_t_sca, c_t_mea, c_t_goal, c_t_gm = cur_data

                # 获取新的衡量 d_mea
                c_mea[_constants.reach_goals] += achive_count

                if c_done:  # 如果本步结束
                    f_index += 1
                    achive_count = 0
                    data = (c_img, c_sca, c_mea, c_goal, c_gm, c_action, c_rew, c_done, c_win,
                            c_t_img, c_t_sca, c_t_mea, c_t_goal, c_t_gm)
                    self.her_storage.append(data)
                else:
                    # 获取f goalmap
                    _, _, f_mea, _, f_gm, _, _, _, _, _, _, _, _, _ = f_data[f_index]
                    og_fgm = np.stack(f_gm, axis=2)
                    og_fgm = np.stack(og_fgm, axis=2)
                    f_idx = og_fgm[_idx_map]
                    # 获取current goalmap
                    og_gm = np.stack(c_gm, axis=2)
                    og_gm = np.stack(og_gm, axis=2)
                    # [passage, rigid, img['idx'], extra_bomb]
                    # 获取real_gm
                    og_gm[_goal_map] = f_idx
                    c_gm = np.stack(og_gm, axis=2)
                    # 获取下一步
                    og_tgm = np.stack(c_t_gm, axis=2)
                    og_tgm = np.stack(og_tgm, axis=2)

                    # 如果下一步到达目标点
                    if np.logical_and(og_gm[_goal], og_tgm[_idx_map]).any():
                        achive_count += 1

                    data = (c_img, c_sca, c_mea, c_goal, c_gm, c_action, c_rew, c_done, c_win,
                            c_t_img, c_t_sca, c_t_mea, c_t_goal, c_t_gm)
                    self.her_storage.append(data)

            while len(self.her_storage) > self.maxsize:
                self.her_storage.pop(0)  # 弹出buffer第一位

    def sample(self):
        return self.seq_sample()

    def seq_sample(self):
        if self.hindsight and random.random() < _constants.her_pb:
            storage = self.her_storage
        else:
            storage = self.storage
        _len = len(storage) - 1 - self.time_spans[-1]
        imgs, scas, meas, goals, gms, actions, futures = [], [], [], [], [], [], []
        # terminal = True
        # idx = random.randint(0, _len)
        for i in range(self.batch_size):
            idx = random.randint(0, _len)

            data = storage[idx]
            img, sca, mea, goal, gm, action, rew, done, win, t_img, t_sca, t_mea, t_goal, t_gm = data
            imgs.append(img)
            scas.append(sca)
            meas.append(mea)
            goals.append(goal)
            gms.append(gm)
            actions.append(action)
            future = self.compute_future(idx=idx, storage=storage)
            futures.append(future)

        return np.array(imgs), np.array(scas), np.array(meas), np.array(goals), np.array(gms), \
               np.array(actions), np.array(futures)

    def compute_future(self, idx, storage):
        future = []
        cur_mea = storage[idx][2]
        j = idx
        terminal = False
        # print('~~~~~~~~~~~~~~~~~~~~~~')
        while j - idx <= self.time_spans[-1]:
            _, _, j_mea, _, _, _, _, done, _, \
            _, _, _, _, _ = storage[j]
            if done and not terminal:  # 代表结束
                terminal = True
                terminal_mea = j_mea
            if (j - idx) in self.time_spans:  # 如果在 timespans 内
                if terminal:
                    diff_mea = terminal_mea - cur_mea
                else:
                    diff_mea = j_mea - cur_mea
                future.extend(diff_mea)

                print('j-idx:', j - idx)
                if terminal:
                    print("terminal_mea:", terminal_mea)
                else:
                    print("j_mea:", j_mea)
                print("cur_mea:", cur_mea)
                print('-=-=-=-=-=-=-diff_mea', diff_mea)

            j += 1
        return np.array(future)
