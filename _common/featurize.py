import copy
import queue
import numpy as np
from pommerman import constants, utility
from gym import spaces
import random
from . import _constants

passage = constants.Item.Passage.value
rigid = constants.Item.Rigid.value
wood = constants.Item.Wood.value
item = constants.Item.ExtraBomb.value
fog = constants.Item.Fog.value
extra_bomb = constants.Item.ExtraBomb.value
incr_range = constants.Item.IncrRange.value
kick = constants.Item.Kick.value


def get_img_space():
    return spaces.Box(low=0, high=1, shape=(11, 11, 9))


# def get_goalmap_space():
#     return spaces.Box(low=0, high=1, shape=(11, 11, 4))


def get_scas_space():
    return spaces.Box(low=0, high=1, shape=(7,))
    # 7dim: [steps, ammo, strength, kick, teammate, enemy1, enemy2]


def get_meas_space():
    return spaces.Box(low=0, high=1, shape=(_constants.meas_size,))
    # 7dim: [woods↑, items↑, ammo_used↑, frags↑, is_dead↑, imove_counts↑]


def get_goal_space():
    return get_meas_space()


def get_action_space():
    return spaces.Discrete(_constants.n_actions)


# 分离img \ meas \ scalars 并处理为网络的输入
def featurize(obs):
    # 直接使用原来obs作为 img
    img = copy.deepcopy(obs)
    img['board'] = board_abstract(img['board'])
    img['bomb_map'] = get_all_bomb_map(img)
    img_fea = img_extra(img)

    # 提取标量
    scas = dict()
    scas['step_count'] = obs['step_count']
    scas['blast_strength'] = obs['blast_strength']
    scas['can_kick'] = obs['can_kick']
    scas['alives'] = obs['alive']
    scas['teammate'] = obs['teammate']
    scas['enemies'] = obs['enemies']
    scas['ammo'] = obs['ammo']
    scas_fea = scalars_extra(scas)

    # 提取衡量指标
    meas = dict()
    meas['items'] = obs['items']
    meas['ammo_used'] = obs['ammo_used']
    meas['woods'] = obs['woods']
    meas['frags'] = obs['frags']
    meas['is_dead'] = obs['is_dead']
    meas['position'] = obs['position']
    # meas['goal_positions'] = obs['goal_positions']
    # meas['reach_goals'] = obs['reach_goals']
    meas['step_counts'] = obs['step_count']
    meas['imove_counts'] = obs['imove_counts']
    meas_fea = measurements_extra(meas)

    # 提取 goal
    goal_fea = np.array(obs['goal'], dtype=np.float32)

    # 提取 goalmap
    # gm_fea = goalmap_extra(img)
    # print('imove_counts', obs['imove_counts'])
    # print('step_count', obs['step_count'])
    return img_fea, scas_fea, meas_fea, goal_fea  # , gm_fea
    # [ (11, 11, 10), (7, ), (5, ), (5, ), (11, 11, 3) ]


# 状态抽象
def board_abstract(board):
    # 将 items 处理为相同编号
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[(r, c)] in [extra_bomb, incr_range, kick]:
                board[(r, c)] = extra_bomb

    return board


# 特征提取 one-hot
def img_extra(img):
    board = img['board']
    bomb_map = img['bomb_map']
    move_direction = img['bomb_moving_direction']
    train_idx = img['idx']
    teammate_idx = img['teammate'].value
    enemies_idx = []
    for e in img['enemies']:
        if not e.value == 9:  # AgentDummy
            enemies_idx.append(e.value)

    maps = []
    for i in [fog, passage, rigid, wood, item, train_idx, teammate_idx]:
        maps.append(board == i)
    maps.append(np.logical_or(
        board == enemies_idx[0], board == enemies_idx[1]))

    maps.append(bomb_map / 12)
    # maps.append(move_direction / 4)

    return np.array(np.stack(maps, axis=2), dtype=np.float32)  # 11 * 11 * 9


# goalmap提取
# def goalmap_extra(img):
#     board = img['board']
#     maps = []
#     for i in [passage, rigid, img['idx'], extra_bomb]:
#         maps.append(board == i)
#
#     return np.array(np.stack(maps, axis=2), dtype=np.float32)  # 11 * 11 * 4


# 标量提取
def scalars_extra(scas):
    maps = []

    step = scas['step_count']  # / 800 if scas['step_count'] / 800 <= 1 else 1
    ammo = scas['ammo']  # / 4 if scas['ammo'] / 4 <= 1 else 1
    blast_strength = scas['blast_strength']  # / 6 if scas['blast_strength'] / 6 <= 1 else 1

    maps.append(step)
    maps.append(ammo)
    maps.append(blast_strength)
    maps.append(scas['can_kick'])

    teammate = scas['teammate'].value
    enemies = []
    for e in scas['enemies']:
        if not e.value == 9:  # AgentDummy
            enemies.append(e.value)

    for aliv in [teammate, enemies[0], enemies[1]]:
        a = 1 if aliv in scas['alives'] else 0
        maps.append(a)

    return np.array(maps, dtype=np.float32)
    # 7 -> [steps, ammo, strength, kick, teammate, enemy1, enemy2]


# 衡量指标提取
# 7dim: [woods↑, items↑, ammo_used↑, frags↑, is_dead↑, reach_goals↑, imove_counts↑]
def measurements_extra(meas):
    maps = []
    woods = meas['woods']  # / 15 if meas['woods'] / 15 <= 1 else 1
    items = meas['items']  # / 10 if meas['items'] / 10 <= 1 else 1
    ammo_used = meas['ammo_used']  # / 20 if meas['ammo_used'] / 20 <= 1 else 1

    maps.append(woods)
    maps.append(items)
    maps.append(ammo_used)
    maps.append(meas['frags'])
    maps.append(meas['is_dead'])

    # maps.append(meas['reach_goals'])
    maps.append(meas['imove_counts'])
    return np.array(maps, dtype=np.float32)


# 提取特定位置 position_bomb_map
def get_position_bomb_map(bomb_map, position, rang=11):
    q = queue.Queue()
    q.put(position)
    used_position = []
    position_bomb_map = np.zeros(shape=(rang, rang))
    position_bomb_life = bomb_map[position]
    if position_bomb_life > 0:
        while not q.empty():
            p = q.get()
            p_x, p_y = p
            # up, down, left, right
            # [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for act_toward in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                x = act_toward[0] + p_x
                y = act_toward[1] + p_y
                if (x, y) not in used_position and 0 <= x <= rang - 1 and 0 <= y <= rang - 1:
                    if bomb_map[(x, y)] == position_bomb_life:
                        q.put((x, y))
                        position_bomb_map[(x, y)] = position_bomb_life
                        used_position.append((x, y))
    return position_bomb_map


# 提取全局 bomb_map
def get_all_bomb_map(img, rang=11):
    board = copy.deepcopy(img['board'])
    bomb_life = copy.deepcopy(img['bomb_life'])
    bomb_blast_strength = copy.deepcopy(img['bomb_blast_strength'])
    flame_life = copy.deepcopy(img['flame_life'])

    # 统一炸弹时间
    for x in range(rang):
        for y in range(rang):
            if bomb_blast_strength[(x, y)] > 0:
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x + i, y)
                    if x + i > rang - 1:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break

                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x - i, y)
                    if x - i < 0:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x, y + i)
                    if y + i > rang - 1:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x, y - i)
                    if y - i < 0:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]

    bomb_life = np.where(bomb_life > 0, bomb_life + 3, bomb_life)
    flame_life = np.where(flame_life == 0, 15, flame_life)
    # flame_life = np.where(flame_life == 1, 15, flame_life)
    bomb_life = np.where(flame_life != 15, flame_life, bomb_life)

    return bomb_life


def extra_position(pos_abs, state):
    if pos_abs == 121:
        return state['position']

    for x in range(11):
        for y in range(11):
            if x * 11 + y == pos_abs:
                return (x, y)


def position_is_passable(board, position, enemies):
    '''Determins if a possible can be passed'''
    return all([
        any([
            utility.position_is_agent(board, position),
            utility.position_is_powerup(board, position),
            utility.position_is_passage(board, position),
            utility.position_is_fog(board, position),
        ]), not utility.position_is_enemy(board, position, enemies)
    ])


def isLegal_act(state, move, rang=11):
    my_x, my_y = state['position']
    row, col = move
    passage = constants.Item.Passage.value
    bomb = constants.Item.Passage.value
    if 0 <= my_x + row <= rang - 1 and 0 <= my_y + col <= rang - 1:
        if state['can_kick']:
            return state['board'][(my_x + row, my_y + col)] in [bomb, passage]
        else:
            return state['board'][(my_x + row, my_y + col)] == passage
    else:
        return False


def dijkstra_act(obs_nf, goal_abs, exclude=None, rang=11):
    # 放炸弹
    if goal_abs == rang * rang:
        return 5

    # 停止在原地
    my_position = tuple(obs_nf['position'])
    goal = extra_position(goal_abs, obs_nf)
    if goal == my_position:
        return 0

    board = np.array(obs_nf['board'])
    enemies = [constants.Item(e) for e in obs_nf['enemies']]

    if exclude is None:
        exclude = [
            constants.Item.Rigid,
            constants.Item.Wood,
        ]

    dist = {}
    prev = {}
    Q = queue.Queue()

    # my_x, my_y = my_position
    for r in range(0, rang):
        for c in range(0, rang):
            position = (r, c)

            if any([utility.position_in_items(board, position, exclude)]):
                continue

            prev[position] = None

            if position == my_position:
                Q.put(position)
                dist[position] = 0
            else:
                dist[position] = np.inf

    while not Q.empty():
        position = Q.get()

        if position_is_passable(board, position, enemies):
            x, y = position
            val = dist[(x, y)] + 1
            for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_position = (row + x, col + y)
                if new_position not in dist:
                    continue

                if val < dist[new_position]:
                    dist[new_position] = val
                    prev[new_position] = position
                    Q.put(new_position)
                elif val == dist[new_position] and random.random() < .5:
                    dist[new_position] = val
                    prev[new_position] = position

    row_g, col_g = goal
    my_x, my_y = my_position
    up = (-1, 0)
    down = (1, 0)
    left = (0, -1)
    right = (0, 1)
    # 判断goal是否可以达到
    while goal in dist and prev[goal] != my_position:
        goal = prev[goal]

    legal_act = []
    # 无法到达有效目的
    if goal not in dist:
        # 可以向下行走
        if row_g > my_x:
            if isLegal_act(obs_nf, down, rang=rang): legal_act.append(2)
        elif row_g < my_x:
            if isLegal_act(obs_nf, up, rang=rang): legal_act.append(1)
        # 可以向右行走
        if col_g > my_x:
            if isLegal_act(obs_nf, right, rang=rang): legal_act.append(4)
        elif col_g < my_x:
            if isLegal_act(obs_nf, left, rang=rang): legal_act.append(3)
        if legal_act:
            return random.choice(legal_act)
    # 可以达到的目的
    else:
        count = 1
        for act_to in [up, down, left, right]:
            row, col = act_to
            if goal == (my_x + row, my_y + col):
                return count
            count += 1
    return 0
