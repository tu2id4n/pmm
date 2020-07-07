import pommerman
from pommerman import agents
from tqdm import tqdm
from _pommerman import _agents
from _common import featurize
from _baselines import DFP

agent_list = [
    # agents.DockerAgent('multiagentlearning/hakozakijunctions', port=1021),
    # agents.SimpleAgent(),
    _agents.SuicideAgent(),
    agents.SimpleAgent(),
    agents.SimpleAgent(),
    agents.SimpleAgent(),
]
env = pommerman.make('PommeRadioCompetition-v21', agent_list)

model = DFP.load(load_path="model/test_600k.zip")
train_idx = 0
for episode in tqdm(range(1000)):
    # [woods, items, ammo_used, frags, is_dead]
    obs = env.reset(train_idx=train_idx, goal=[0.1, 0.9, 0.5, 0, 0.1])
    done = False
    while not done:
        all_actions = env.act(obs)
        print("primitive", all_actions[train_idx])
        featurize_obs = featurize.featurize(obs[train_idx])
        print(featurize_obs[2])
        train_act = model.predict(featurize_obs)
        all_actions[train_idx] = train_act
        print("train", all_actions[train_idx])
        obs, rewards, done, info = env.step(all_actions)
        env.render()
    print(info, rewards)
print('1000 test ok')

env.close()
