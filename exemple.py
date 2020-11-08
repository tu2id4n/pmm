import pommerman
from pommerman import agents
from _pommerman import _agents
from tqdm import tqdm
from _common import featurize, _constants

agent_list = [
    _agents.RandAgent(),
    _agents.SuicideAgent(),
    _agents.SuicideAgent(),
    _agents.SuicideAgent(),
]

env = pommerman.make('maze-v1', agent_list)

red_win = 0
for episode in tqdm(range(100)):
    obs = env.reset()
    done = False
    count = 0
    woods = 0
    frags = 0
    a = 0
    while not done:
        env.render()
        all_actions = env.act(obs)
        # fea = featurize.featurize(obs[0])
        # print("random act:", all_actions[0])
        # print('dijk_step:', env.dijk_step)

        if env.is_dijk or env.dijk_step > _constants.max_dijk:
            a = input("input:")
        all_actions[0] = int(a)
        # print(all_actions[0])

        obs, rewards, done, info = env.step(all_actions)

        print(obs[0]['items'])
        print(rewards)
        # print("imove_counts", obs[0]['imove_counts'])

        # print(obs[0]['my_bomb'])
        # if obs[0]['woods'] > woods:
        #     woods = obs[0]['woods']
        #     print('woods', obs[0]['woods'])

        # if obs[0]['frags'] > frags:
        #     frags = obs[0]['frags']
        #     print("frags", obs[0]['frags'])

        # print('reach:', obs[0]['reach'])
        # env.render()
    print(rewards, info)
    if rewards[0] == 1:
        red_win += 1
print('1000 test:', red_win)
env.close()
