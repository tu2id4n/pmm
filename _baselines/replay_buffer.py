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
    

    def add(self, obs, actions, rews, dones, terminal_obs, wins):
        imgs, scas, meas, goals = obs
        t_imgs, t_scas, t_meas, t_goals = terminal_obs
        data = (imgs, scas, meas, goals, actions, rews, dones, wins, t_imgs, t_scas, t_meas, t_goals)
        self.storage.append(data)
        while len(self.storage) > self.maxsize:
            self.storage.pop(0)

    def sample(self):
        idxes = [random.randint(0, len(self.storage) - 1 - self.time_spans[-1]) for _ in range(self.batch_size)]
        imgs, scas, meas, goals, actions, futures = [], [], [], [], [], []
        for i in idxes:
            data = self.storage[i]
            img, sca, mea, goal, action, rew, done, win, t_img, t_sca, t_mea, t_goal = data
            imgs.append(img)
            scas.append(sca)
            meas.append(mea)
            goals.append(goal)
            actions.append(action)
            futures.append(self.compute_future(idx=i))
        
        return np.array(imgs), np.array(scas), np.array(meas), np.array(goals), np.array(actions), np.array(futures)

    def compute_future(self, idx):
        future = []
        cur_mea = self.storage[idx][2]
        j = idx + 1
        terminal = False
        while j - idx <= self.time_spans[-1]:
            if (j - idx) in self.time_spans:
                img, sca, mea, goal, action, rew, done, win, t_img, t_sca, t_mea, t_goal = self.storage[j]
                if done and not terminal:
                    terminal = True
                    terminal_mea = t_mea
                if terminal:
                    future.extend(terminal_mea - cur_mea)
                else:
                    future.extend(mea - cur_mea)
            j += 1

        return np.array(future)