import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, time_spans, batch_size):
        self.storage = []
        self.maxsize = buffer_size
        self.time_spans = time_spans
        self.future_size = len(time_spans)
        self.batch_size = batch_size

    def can_sample(self):
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self.storage) >= 2 * self.batch_size

    def add(self, obs, actions, rews, dones, terminal_obs, wins, hindsight=False):
        imgs, scas, meas, goals, gms = obs
        t_imgs, t_scas, t_meas, t_goals, t_gms = terminal_obs
        data = (imgs, scas, meas, goals, gms, actions, rews, dones,
                wins, t_imgs, t_scas, t_meas, t_goals, t_gms)
        self.storage.append(data)
        while len(self.storage) > self.maxsize:
            self.storage.pop(0)

        if hindsight and random.random() < 0.5:
            new_imgs = np.stack(imgs, axis=2)
            new_imgs = np.stack(new_imgs, axis=2)
            
            new_timgs = np.stack(t_imgs, axis=2)
            new_timgs = np.stack(new_timgs, axis=2)
            
            new_gms = np.stack(gms, axis=2)
            new_gms = np.stack(new_gms, axis=2)
           
            new_gms[2] = np.logical_or(new_imgs[5], new_timgs[5])
            new_gms = np.stack(new_gms, axis=2)
           
            new_tmeas = t_meas
            new_tmeas[-2] = True
    
            data = (imgs, scas, meas, goals, new_gms, actions, rews, dones, wins,
                    t_imgs, t_scas, new_tmeas, t_goals, new_gms)

        self.storage.append(data)
        while len(self.storage) > self.maxsize:
            self.storage.pop(0)

    def sample(self):
        return self.seq_sample()

    def seq_sample(self):
        _len = len(self.storage) - 1 - self.time_spans[-1]
        imgs, scas, meas, goals, gms, actions, futures = [], [], [], [], [], [], []
        terminal = True
        for i in range(self.batch_size):
            if terminal:
                idx = random.randint(0, _len)

            data = self.storage[idx]
            img, sca, mea, goal, gm, action, rew, done, win, t_img, t_sca, t_mea, t_goal, t_gm = data
            imgs.append(img)
            scas.append(sca)
            meas.append(mea)
            goals.append(goal)
            gms.append(gm)
            actions.append(action)
            future, terminal = self.compute_future(idx=idx)
            futures.append(future)

            # 获得连续的经验.
            idx += 1

            # 如果idx超出界限, 则需要重选idx.
            if idx > _len:
                terminal = True

        return np.array(imgs), np.array(scas), np.array(meas), np.array(goals), np.array(gms), \
               np.array(actions), np.array(futures)

    def compute_future(self, idx):
        future = []
        cur_mea = self.storage[idx][2]
        j = idx
        terminal = False
        while j - idx <= self.time_spans[-1]:
            img, sca, mea, goal, gm, action, rew, done, win, t_img, t_sca, t_mea, t_goal, t_gm = self.storage[
                j]
            if done:  # 代表结束
                terminal = True
                terminal_mea = t_mea

            if (j - idx) in self.time_spans:  # 如果在 timespans 内
                if terminal:
                    future.extend(terminal_mea - cur_mea)
                else:
                    future.extend(mea - cur_mea)
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
