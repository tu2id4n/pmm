import pommerman
from pommerman import agents
from _pommerman import _agents


def make_envs(env_id):
    print('env = ', env_id)

    def _thunk():
        agent_list = [
            agents.SimpleAgent(),
            _agents.SuicideAgent(),
            _agents.SuicideAgent(),
            _agents.SuicideAgent(),

            # _agents.StopAgent(),
            # _agents.SuicideAgent(),
            # hit18Agent('1'),
            # hit18Agent('3')
        ]
        env = pommerman.make(env_id, agent_list)
        return env

    return _thunk