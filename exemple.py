import pommerman
from pommerman import agents
from _pommerman import _agents
from tqdm import tqdm
from _common import featurize

agent_list = [
    _agents.SuicideAgent(),
    _agents.StopAgent(),
    _agents.StopAgent(),
    _agents.StopAgent(),
]

env = pommerman.make('maze-v2', agent_list)

red_win = 0
for episode in tqdm(range(100)):
    obs = env.reset()
    done = False
    count = 0
    while not done:
        all_actions = env.act(obs)
        # fea = featurize.featurize(obs[0])
        # print("random act:", all_actions[0])
        obs, rewards, done, info = env.step(all_actions)
        print("imove_counts", obs[0]['imove_counts'])
        env.render()
    print(rewards, info)
    if rewards[0] == 1:
        red_win += 1
print('1000 test:', red_win)
env.close()
