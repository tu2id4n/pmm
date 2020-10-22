'''An agent that preforms a random action each step'''
from pommerman.agents import BaseAgent
from pommerman import characters


class SuicideAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def act(self, obs, action_space):
        self._character.reset(blast_strength=1)
        return 5
