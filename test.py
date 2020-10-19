import pommerman
from pommerman import agents
from tqdm import tqdm
from _pommerman import _agents
from _common import featurize
from _baselines import DFP
import random
import numpy as np

_test = False
_env_name = 'maze-v1'
_model_path = "model/maze1_dfp_her_9M.zip"
# 8dim: [woods↑, items↑, ammo_used↑↓, frags↑, is_dead↑, reach_goals↑, step_counts↑, imove_counts↑]
_goal = [0, 0, 0, 0, 1, 0, -1]
_train_idx = 0
_episode = 1000
_flag = False


class Test:
    def __init__(self):
        self.agent_list = [
            agents.SimpleAgent(),
            _agents.SuicideAgent(),
            _agents.SuicideAgent(),
            _agents.SuicideAgent(),
        ]
        self.env = pommerman.make(_env_name, self.agent_list)
        self.train_idx = _train_idx
        self.goal = _goal
        self.flag = _flag
        self.episode = _episode

    def run(self):
        self.model = DFP.load(load_path=_model_path)
        print("Press D in game window to switch to next game episode")
        for episode in range(self.episode):
            obs = self.env.reset(train_idx=self.train_idx, goal=self.goal)
            done = False
            first_render = True
            while not done:
                all_actions = self.env.act(obs)
                featurize_obs = featurize.featurize(obs[self.train_idx])
                train_act = self.model.predict(featurize_obs)
                print("train_act:", train_act)

                all_actions[self.train_idx] = train_act
                obs, rewards, done, info = self.env.step(all_actions)
                self.env.render()

                if first_render:  # 第一次 render env 时将当前的测试注册.
                    self.env._viewer.window.push_handlers(self)

                if self.flag:
                    self.flag = False
                    done = True

            print(info)
            print()

        print('1000 test ok')
        self.env.close()

    def example(self):
        print("Press D in game window to switch to next game episode")
        for episode in range(self.episode):
            obs = self.env.reset(train_idx=self.train_idx, goal=self.goal)
            done = False
            first_render = True
            while not done:
                print(obs[0])
                all_actions = self.env.act(obs)
                obs, rewards, done, info = self.env.step(all_actions)
                # print(obs[0]['goal_positions'])
                self.env.render()

                if first_render:  # 第一次 render env 时将当前的测试注册.
                    self.env._viewer.window.push_handlers(self)

                if self.flag:
                    self.flag = False
                    done = True

            print(info)
            print()

        print('1000 test ok')
        self.env.close()

    def on_key_press(self, symbol, mod):
        from pyglet.window import key
        if symbol == key.D:
            self.flag = True


if __name__ == '__main__':
    if _test:
        Test().run()
    else:
        Test().example()
