import pommerman
from pommerman import agents
from tqdm import tqdm
from _pommerman import _agents
from _common import featurize
from _baselines import DFP

global Flag
Flag = False


class Test:
    def __init__(self):
        self.agent_list = [
            agents.SimpleAgent(),
            _agents.SuicideAgent(),
            _agents.SuicideAgent(),
            _agents.SuicideAgent(),
        ]

        self.env = pommerman.make('maze-v1', self.agent_list)
        self.model = DFP.load(load_path="model/maze1_dfp_7M.zip")
        self.train_idx = 0
        #  7 -> [woods, items, ammo_used, frags, is_dead, reach_goal, step]
        self.goal = [0, 0, -1, 0, -1, 1, -0.05]
        self.flag = False
        self.episode = 1000

    def run(self):
        print("Press D in game window to switch to next game episode")
        for episode in range(self.episode):
            obs = self.env.reset(train_idx=self.train_idx, goal=self.goal)
            print('curr goal', self.env.goal)
            done = False
            first_render = True
            while not done:
                all_actions = self.env.act(obs)
                featurize_obs = featurize.featurize(obs[self.train_idx])
                train_act = self.model.predict(featurize_obs)
                all_actions[self.train_idx] = train_act
                print("train_act:", train_act)

                obs, rewards, done, info = self.env.step(all_actions)
                self.env.render()

                if first_render:  # 第一次 render env 时将当前的测试注册.
                    self.env._viewer.window.push_handlers(self)

                if self.flag:
                    self.flag = False
                    done = True
                # if done:
                #     print('dead')
                #     self.env._agents[self.train_idx + 2].is_alive = True

            print(info)
            print()

        print('1000 test ok')
        self.env.close()

    def on_key_press(self, symbol, mod):
        from pyglet.window import key
        if symbol == key.D:
            self.flag = True


if __name__ == '__main__':
    Test().run()
