import pommerman
from pommerman import agents
from _pommerman import _agents

agent_list = [
    # _agents.StopAgent(),
    agents.SimpleAgent(),
    _agents.SuicideAgent(),
    _agents.SuicideAgent(),
    _agents.SuicideAgent(),
]

env = pommerman.make('maze-v1', agent_list)

for episode in range(10000):
    obs = env.reset()
    done = False
    first_render = True
    count = 0
    while not done:
        all_actions = env.act(obs)
        obs, rewards, done, info = env.step(all_actions)
        print(rewards[0])
        env.render()

        # if done and not count:
        #     count += 1
        #     env._agents[2]._character.is_alive = True;  # reborn
    print(info)
print('1000 test ok')
env.close()
