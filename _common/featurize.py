import copy
import queue
import numpy as np
from pommerman import constants
from gym import spaces

passage = constants.Item.Passage.value
rigid = constants.Item.Rigid.value
wood = constants.Item.Wood.value
item = constants.Item.ExtraBomb.value
fog = constants.Item.Fog.value
extra_bomb = constants.Item.ExtraBomb.value
incr_range = constants.Item.IncrRange.value
kick = constants.Item.Kick.value


def get_img_space():
    return spaces.Box(low=0, high=1, shape=(11, 11, 10))


def get_scas_space():
    return spaces.Box(low=0, high=1, shape=(7,))


def get_meas_space():
    return spaces.Box(low=0, high=1, shape=(5,))


def get_goal_space():
    return spaces.Box(low=-1, high=1, shape=(5,))


def get_action_space():
    return spaces.Discrete(6)


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
    meas_fea = measurements_extra(meas)

    # 提取目标
    goal_fea = np.array(obs['goal'])

    return img_fea, scas_fea, meas_fea, goal_fea  # [ (11, 11, 10), (7, ), (5, ), (5, ) ]


# 状态抽象
def board_abstract(board):
    # 将 items 处理为相同编号
    for r in range(len(board)):
        for c in range(len(board[0])):
            if (board[(r, c)] in [extra_bomb, incr_range, kick]):
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
    for i in [passage, rigid, wood, item, fog, train_idx, teammate_idx]:
        maps.append(board == i)
    maps.append(np.logical_or(
        board == enemies_idx[0], board == enemies_idx[1]))

    maps.append(bomb_map / 13)
    maps.append(move_direction / 4)

    return np.stack(maps, axis=2)  # 11 * 11 * 10


# 标量提取
def scalars_extra(scas):
    maps = []

    ammo = scas['ammo'] / 5 if scas['ammo'] / 5 <= 1 else 1
    blast_strength = scas['blast_strength'] / 6 if scas['blast_strength'] / 6 <= 1 else 1

    maps.append(scas['step_count'] / 801)
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

    return np.array(maps)  # 7 -> [step, ammo, strength, kick, teammate, enemy1, enemy2]


# 衡量指标提取
def measurements_extra(meas):
    maps = []
    woods = meas['woods'] / 10 if meas['woods'] / 10 <= 1 else 1
    items = meas['items'] / 6 if meas['items'] / 6 <= 1 else 1
    ammo_used = meas['ammo_used'] / 50 if meas['ammo_used'] / 50 <= 1 else 1

    maps.append(woods)
    maps.append(items)
    maps.append(ammo_used)
    maps.append(meas['frags'] / 2)
    maps.append(meas['is_dead'])

    return np.array(maps)  # 4 -> [woods, items, ammo_used, frags, is_dead]


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
    flame_life = np.where(flame_life == 1, 15, flame_life)
    bomb_life = np.where(flame_life != 15, flame_life, bomb_life)

    return bomb_life
