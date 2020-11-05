import pommerman
from pommerman import agents
from tqdm import tqdm
from _pommerman import _agents
from _common import featurize
from _baselines import DFP
import random
import numpy as np
from _common import _constants

_test = True  # False to example
_model_path = "model/test_100k.zip"
_env_name = 'maze-v2'  # 'PommeRadioCompetition-v21'
_episode = 1000
_flag = False
_agent_list = [agents.SimpleAgent(),
               agents.SimpleAgent(),
               agents.SimpleAgent(),
               agents.SimpleAgent(),
               ]


class Test:
    def __init__(self):
        self.agent_list = _agent_list
        self.env = pommerman.make(_env_name, self.agent_list)
        self.train_idx = _constants.train_idx
        self.goal = _constants.train_goal
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

                # if random.random() < 0.2:
                #     all_actions[self.train_idx] = np.array([random.randint(1, 4)])
                # else:
                all_actions[self.train_idx] = train_act

                obs, rewards, done, info = self.env.step(all_actions)

                # print(obs[self.train_idx]['position'], '->', featurize.extra_position(train_act, obs[self.train_idx]))
                self.env.render()

                if first_render:  # 第一次 render env 时将当前的测试注册.
                    self.env._viewer.window.push_handlers(self)

                if self.flag:
                    self.flag = False
                    done = True
            print("reward:", rewards[0])
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
                all_actions = self.env.act(obs)
                obs, rewards, done, info = self.env.step(all_actions)
                # featurize_obs = featurize.featurize(obs[self.train_idx])

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
