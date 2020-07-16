import pommerman
from pommerman import agents
from _pommerman import _agents

agent_list = [
    _agents.StopAgent(),
    agents.SimpleAgent(),
    _agents.SuicideAgent(),
    agents.SimpleAgent(),
]

env = pommerman.make('PommeRadioCompetition-v21', agent_list)

for episode in range(10000):
    obs = env.reset()
    done = False
    first_render = True
    count = 0
    while True:
        all_actions = env.act(obs)
        obs, rewards, done, info = env.step(all_actions)
        env.render()
        print(obs[2]['step_count'])
        if not env._agents[2].is_alive:
            print(obs[2]['step_count'])

        # if done and not count:
        #     count += 1
        #     env._agents[2]._character.is_alive = True;  # reborn
    print(info)
    print()

print('1000 test ok')
env.close()
