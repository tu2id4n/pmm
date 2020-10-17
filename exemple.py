import pommerman
from pommerman import agents
from _pommerman import _agents
from tqdm import tqdm

agent_list = [
    agents.SimpleAgent(),
    _agents.SuicideAgent(),
    _agents.SuicideAgent(),
    _agents.SuicideAgent(),
]

env = pommerman.make('maze-v1', agent_list)

for episode in tqdm(range(10000)):
    obs = env.reset(goal=[0, 0, -1, 0, -1, 1, -0.05])
    done = False
    first_render = True
    count = 0
    while not done:
        all_actions = env.act(obs)
        obs, rewards, done, info = env.step(all_actions)
        env.render()
        print(obs[0])
    print(rewards, info)
print('1000 test ok')
env.close()
