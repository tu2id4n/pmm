import pommerman
from pommerman import agents
from tqdm import tqdm
from _common import featurize
from _baselines import DFP

agent_list = [
    # agents.DockerAgent('multiagentlearning/hakozakijunctions', port=1021),
    agents.SimpleAgent(),
    agents.SimpleAgent(),
    agents.SimpleAgent(),
    agents.SimpleAgent(),
]
env = pommerman.make('PommeRadioCompetition-v21', agent_list)

model = DFP.load(load_path="model/_0k.zip")
train_idx = 0
for episode in tqdm(range(1000)):
    obs = env.reset()
    done = False
    while not done:
        all_actions = env.act(obs)

        print()
        print("primitive", all_actions)
        featurize_obs = featurize.featurize(obs[train_idx])
        train_act = model.predict(featurize_obs)
        all_actions[train_idx] = train_act
        print("train", all_actions)

        obs, rewards, done, info = env.step(all_actions)
        env.render()

    print(info)
print('1000 test ok')

env.close()