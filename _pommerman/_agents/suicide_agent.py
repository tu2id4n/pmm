'''An agent that preforms a random action each step'''
from pommerman.agents import BaseAgent
from pommerman import characters

class SuicideAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""
    # def __init__(self, character=characters.Bomber):
    #     character.blast_strength = 1
    #     self._character = character

    def act(self, obs, action_space):
        return 5
