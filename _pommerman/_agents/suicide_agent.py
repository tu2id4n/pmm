'''An agent that preforms a random action each step'''
from pommerman.agents import BaseAgent


class SuicideAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def act(self, obs, action_space):
        return 5
