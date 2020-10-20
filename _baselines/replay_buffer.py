import random
import numpy as np

_hindsight = False

class ReplayBuffer(object):
    def __init__(self, buffer_size, time_spans, batch_size):
        self.storage = []
        self.maxsize = buffer_size
        self.time_spans = time_spans
        self.future_size = len(time_spans)
        self.batch_size = batch_size
        self.her_size = 15
        self.K = 4
        self.her = 0
        self.her_storage = []
        self.hindsight = _hindsight
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
        if self.hindsight and self.her > self.her_size:
            self.her = 0
            st = len(self.storage) - 1 - self.her_size
            f_data = list(self.storage[st + self.her_size])
            _, _, _, _, f_gm, _, _, _, _, _, _, _, _, _ = f_data
            # 获取f goalmap
            new_fgm = np.stack(f_gm, axis=2)
            new_fgm = np.stack(new_fgm, axis=2)

            achive_count = 0
            for t in range(0, self.her_size):
                cur_data = self.storage[st + t]
                d_img, d_sca, d_mea, d_goal, d_gm, d_action, d_rew, d_done, d_win, \
                d_t_img, d_t_sca, d_t_mea, d_t_goal, d_t_gm = cur_data

                # 获取d goalmap
                # [passage, rigid, img['idx'], extra_bomb]
                # 将 f goal map 的 agent position 加入当前 goal map
                new_gm = np.stack(d_gm, axis=2)
                new_gm = np.stack(new_gm, axis=2)
                new_gm[3] = np.logical_or(new_gm[3], new_fgm[2])
                new_gm = np.stack(new_gm, axis=2)

                # 获取下一帧 dt goal map
                new_tgm = np.stack(d_t_gm, axis=2)
                new_tgm = np.stack(new_tgm, axis=2)
                new_tgm[3] = np.logical_or(new_tgm[3], new_fgm[2])
                new_tgm = np.stack(new_tgm, axis=2)

                # 获取新的衡量 d mea
                # 7dim: [woods↑, items↑, ammo_used↑, frags↑, is_dead↑, reach_goals↑, imove_counts↑]
                d_mea[5] += achive_count

                if np.logical_and(new_gm[3], new_tgm[2]).any():
                    achive_count += 1

                d_t_mea[5] += achive_count
                data = (d_img, d_sca, d_mea, d_goal, new_gm, d_action, d_rew, False, d_win,
                        d_t_img, d_t_sca, d_t_mea, d_t_goal, new_tgm)
                self.her_storage.append(data)
                f_data[4] = new_tgm

            f_data[2][5] += achive_count
            f_data[11][5] += achive_count
            f_data[7] = True
            self.her_storage.append(tuple(f_data))

            while len(self.her_storage) > self.maxsize:
                self.her_storage.pop(0)

    def sample(self):
        return self.seq_sample()

    def seq_sample(self):
        if self.hindsight and random.random() < 0.2:
            storage = self.her_storage
        else:
            storage = self.storage
        _len = len(storage) - 1 - self.time_spans[-1]
        imgs, scas, meas, goals, gms, actions, futures = [], [], [], [], [], [], []
        terminal = True
        idx = random.randint(0, _len)
        for i in range(self.batch_size):
            if terminal:
                idx = random.randint(0, _len)

            data = storage[idx]
            img, sca, mea, goal, gm, action, rew, done, win, t_img, t_sca, t_mea, t_goal, t_gm = data
            imgs.append(img)
            scas.append(sca)
            meas.append(mea)
            goals.append(goal)
            gms.append(gm)
            actions.append(action)
            future, terminal = self.compute_future(idx=idx, storage=storage)
            futures.append(future)

            # 获得连续的经验.
            idx += 1

            # 如果idx超出界限, 则需要重选idx.
            if idx > _len:
                terminal = True

        return np.array(imgs), np.array(scas), np.array(meas), np.array(goals), np.array(gms), \
               np.array(actions), np.array(futures)

    def compute_future(self, idx, storage):
        future = []
        cur_mea = storage[idx][2]
        j = idx
        terminal = False
        while j - idx <= self.time_spans[-1]:
            img, sca, mea, goal, gm, action, rew, done, win, \
            t_img, t_sca, t_mea, t_goal, t_gm = storage[j]
            if done:  # 代表结束
                terminal = True
                terminal_mea = mea

            if (j - idx) in self.time_spans:  # 如果在 timespans 内
                if terminal:
                    diff_mea = terminal_mea - cur_mea
                else:
                    diff_mea = mea - cur_mea
                # 设置为贴近 1
                # 7dim: [woods↑, items↑, ammo_used↑, frags↑, is_dead↑, reach_goals↑, imove_counts↑]
                diff_mea[0] = (diff_mea[0] + 1) / 2  # woods↑
                diff_mea[1] = (diff_mea[1] + 1) / 2  # items↑
                diff_mea[2] = (diff_mea[2] + 1) / 2  # ammo_used↑
                diff_mea[3] = (diff_mea[3] + 1) / 2  # frags↑
                diff_mea[4] = (diff_mea[4] + 1) / 2  # is_dead↑
                diff_mea[5] = (diff_mea[5] + 1) / 2  # reach_goals↑
                diff_mea[6] = (diff_mea[6] + 1) / 2  # imove_counts↑
                future.extend(diff_mea)
            j += 1
        return np.array(future), terminal

    def rand_sample(self):
        idxes = [random.randint(0, len(self.storage) - 1 - self.time_spans[-1])
                 for _ in range(self.batch_size)]
        imgs, scas, meas, goals, gms, actions, futures = [], [], [], [], [], [], []

        for i in idxes:
            data = self.storage[i]
            img, sca, mea, goal, gm, action, rew, done, win, t_img, t_sca, t_mea, t_goal, t_gm = data
            imgs.append(img)
            scas.append(sca)
            meas.append(mea)
            goals.append(goal)
            gms.append(gm)
            actions.append(action)
            future, _ = self.compute_future(idx=i)
            futures.append(future)

        return np.array(imgs), np.array(scas), np.array(meas), np.array(goals), np.array(gms), \
               np.array(actions), np.array(futures)
