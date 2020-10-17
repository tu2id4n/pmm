import itertools
import json
import random
import os
from jsonmerge import Merger

from gym import spaces
import numpy as np

from pommerman import constants
from pommerman import utility

def make_board(size, num_rigid=0, num_wood=0, num_agents=4):
    """Make the random but symmetric board.

    The numbers refer to the Item enum in constants. This is:
     0 - passage
     1 - rigid wall
     2 - wood wall
     3 - bomb
     4 - flames
     5 - fog
     6 - extra bomb item
     7 - extra firepower item
     8 - kick
     9 - skull
     10 - 13: agents

    Args:
      size: The dimension of the board, i.e. it's sizeXsize.
      num_rigid: The number of rigid walls on the board. This should be even.
      num_wood: Similar to above but for wood walls.

    Returns:
      board: The resulting random board.
    """

    def lay_wall(value, num_left, coordinates, board):
        '''Lays all of the walls on a board'''
        x, y = random.sample(coordinates, 1)[0]
        coordinates.remove((x, y))
        # coordinates.remove((y, x))
        board[x, y] = value
        # board[y, x] = value
        num_left -= 1
        return num_left

    def lay_item(value, num_left, coordinates, inaccess, board):
        '''Lays all of the walls on a board'''
        x, y = random.sample(coordinates, 1)[0]
        coordinates.remove((x, y))
        while (x, y) in inaccess and len(inaccess) <= len(coordinates):
            x, y = random.sample(coordinates, 1)[0]
            coordinates.remove((x, y))
        board[x, y] = value
        num_left -= 1
        return num_left

    def make(size, num_rigid, num_wood, num_agents):
        '''Constructs a game/board'''
        # Initialize everything as a passage.
        board = np.ones((size,
                         size)).astype(np.uint8) * constants.Item.Passage.value

        # Gather all the possible coordinates to use for walls.
        # x == y 全部是passage
        coordinates = set([
            (x, y) for x, y in \
            itertools.product(range(size), range(size)) \
            if x != y])

        # Set the players down. Exclude them from coordinates.
        # Agent0 is in top left. Agent1 is in bottom left.
        # Agent2 is in bottom right. Agent 3 is in top right.
        assert (num_agents % 2 == 0)

        if num_agents == 2:
            board[1, 1] = constants.Item.Agent0.value
            board[size - 2, size - 2] = constants.Item.Agent1.value
            agents = [(1, 1), (size - 2, size - 2)]
        else:  # 0和2是队友
            board[1, 1] = constants.Item.Agent0.value
            board[size - 2, 1] = constants.Item.Agent1.value
            board[size - 2, size - 2] = constants.Item.Agent2.value
            board[1, size - 2] = constants.Item.Agent3.value
            agents = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]

        for position in agents:
            if position in coordinates:
                coordinates.remove(position)

        # Exclude breathing room on either side of the agents.
        for i in range(2, 4):
            coordinates.remove((1, i))
            coordinates.remove((i, 1))
            coordinates.remove((size - 2, size - i - 1))
            coordinates.remove((size - i - 1, size - 2))

            if num_agents == 4:
                coordinates.remove((1, size - i - 1))
                coordinates.remove((size - i - 1, 1))
                coordinates.remove((i, size - 2))
                coordinates.remove((size - 2, i))
        
        coordinates.remove((1, 0))
        coordinates.remove((0, 1))
        coordinates.remove((size - 2, size- 1))
        coordinates.remove((size - 1, size - 2))

        if num_agents == 4:
            coordinates.remove((1, size - 1))
            coordinates.remove((size - 1, 1))
            coordinates.remove((0, size - 2))
            coordinates.remove((size - 2, 0))

        # Lay down wooden walls providing guaranteed passage to other agents.
        # wood = constants.Item.Wood.value
        # if num_agents == 4:
        #     for i in range(4, size - 4):
        #         board[1, i] = wood
        #         board[size - i - 1, 1] = wood
        #         board[size - 2, size - i - 1] = wood
        #         board[size - i - 1, size - 2] = wood
        #         coordinates.remove((1, i))
        #         coordinates.remove((size - i - 1, 1))
        #         coordinates.remove((size - 2, size - i - 1))
        #         coordinates.remove((size - i - 1, size - 2))
        #         num_wood -= 4

        # Lay down the rigid walls.
        num_rigid = 50
        while num_rigid > 0:
            num_rigid = lay_wall(constants.Item.Rigid.value, num_rigid,
                                 coordinates, board)

        # Lay down the wooden walls.
        # while num_wood > 0:
        #     num_wood = lay_wall(constants.Item.Wood.value, num_wood,
        #                         coordinates, board)

        num_item = 1
        inaccess = utility.inaccessible_passages(board, agents)
        while num_item > 0:
            item_value = random.choice([constants.Item.ExtraBomb, constants.Item.IncrRange,constants.Item.Kick]).value
            num_item = lay_item(item_value, num_item, coordinates, inaccess, board)

        return board, agents

    assert (num_rigid % 2 == 0)
    assert (num_wood % 2 == 0)
    board, agents = make(size, num_rigid, num_wood, num_agents)

    # Make sure it's possible to reach most of the passages.
    while len(utility.inaccessible_passages(board, agents)) > 4:
        board, agents = make(size, num_rigid, num_wood, num_agents)

    return board

    
def make_items(board, num_items):
    '''Lays all of the items on the board'''
    item_positions = {}
    num_items = 0
    while num_items > 0:
        row = random.randint(0, len(board) - 1)
        col = random.randint(0, len(board[0]) - 1)
        # if board[row, col] != constants.Item.Wood.value:
            # continue
        if (row, col) in item_positions:
            continue

        item_positions[(row, col)] = random.choice([
            constants.Item.ExtraBomb, constants.Item.IncrRange,
            constants.Item.Kick
        ]).value
        num_items -= 1
    return item_positions


def generate_item(board, position, size=11):

    def lay_item(value, inaccess, board):
        '''Lays all of the walls on a board'''
        x = random.randint(0,10)
        y = random.randint(0,10)
        while (x, y) in inaccess:
            x = random.randint(0,10)
            y = random.randint(0,10)
        board[x, y] = value
        return board

    agents = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
    inaccess = utility.inaccessible_passages(board, agents)
    inaccess.append(position)
    item_value = random.choice([constants.Item.ExtraBomb, constants.Item.IncrRange,constants.Item.Kick]).value
    board = lay_item(item_value, inaccess, board)
    
    return board