""" maze v1 """
from gym import spaces
import numpy as np

from pommerman import constants
from pommerman import utility
from pommerman.envs import v0
import copy
import random
from . import env_utils

max_setps = 3200
# 7dim: [woods↑, items↑, ammo_used↑, frags↑, is_dead↑, reach_goals↑, imove_counts↑]
meas_size = 7
max_interval = 100


class Pomme(v0.Pomme):
    '''The hardest pommerman environment. This class expands env v0
    adding communication between agents.'''
    metadata = {
        'render.modes': ['human', 'rgb_array', 'rgb_pixel'],
        'video.frames_per_second': constants.RENDER_FPS
    }

    def __init__(self, *args, **kwargs):
        self._radio_vocab_size = kwargs.get('radio_vocab_size')
        self._radio_num_words = kwargs.get('radio_num_words')
        if (self._radio_vocab_size and
            not self._radio_num_words) or (not self._radio_vocab_size and
                                           self._radio_num_words):
            assert ("Include both radio_vocab_size and radio_num_words.")

        self._radio_from_agent = {
            agent: (0, 0)
            for agent in [
            constants.Item.Agent0, constants.Item.Agent1,
            constants.Item.Agent2, constants.Item.Agent3
        ]
        }
        super().__init__(*args, **kwargs)

        self._max_steps = max_setps

    def _set_action_space(self):
        self.action_space = spaces.Tuple(
            tuple([spaces.Discrete(6)] +
                  [spaces.Discrete(self._radio_vocab_size
                                   )] * self._radio_num_words))

    def _set_observation_space(self):
        """The Observation Space for each agent.

        Total observatiosn: 3*board_size^2 + 12 + radio_vocab_size * radio_num_words:
        - all of the board (board_size^2)
        - bomb blast strength (board_size^2).
        - bomb life (board_size^2)
        - agent's position (2)
        - player ammo counts (1)
        - blast strength (1)
        - can_kick (1)
        - teammate (one of {AgentDummy.value, Agent3.value}).
        - enemies (three of {AgentDummy.value, Agent3.value}).
        - radio (radio_vocab_size * radio_num_words)
        """
        bss = self._board_size ** 2
        min_obs = [0] * 3 * bss + [0] * 5 + [constants.Item.AgentDummy.value
                                             ] * 4
        max_obs = [len(constants.Item)] * bss + [self._board_size
                                                 ] * bss + [25] * bss
        max_obs += [self._board_size] * 2 + [self._num_items] * 2 + [1]
        max_obs += [constants.Item.Agent3.value] * 4
        min_obs.extend([0] * self._radio_vocab_size * self._radio_num_words)
        max_obs.extend([1] * self._radio_vocab_size * self._radio_num_words)
        self.observation_space = spaces.Box(
            np.array(min_obs), np.array(max_obs))

    def get_observations(self, ):
        observations = super().get_observations()  # 已经获得新的self.observations了

        for obs in observations:
            obs['message'] = self._radio_from_agent[obs['teammate']]

        observation = observations[self.train_idx]

        # 如果没有obs_pre 就将obs_pre 设置为只当前相同.
        if not self.observation_pre:
            observation['my_bomb'] = []
            observation['woods'] = 0
            observation['frags'] = 0
            observation['items'] = 0
            observation['idx'] = observation['board'][observation['position']]
            observation['ammo_used'] = 0
            observation['goal'] = self.goal
            observation['imove_counts'] = -1
            observation['reach_goals'] = 0
            observation['goal_positions'] = []

            self.observation_pre = observation

        observation['goal'] = self.observation_pre['goal']

        # 通过obs_pre 和obs_now 对比，将my_bomb, woods, frags, items --> obs_now.
        extra_bomb = constants.Item.ExtraBomb.value
        incr_range = constants.Item.IncrRange.value
        kick = constants.Item.Kick.value
        wood = constants.Item.Wood.value
        enemies = []
        for e in observation['enemies']:
            if e.value is not 9:
                enemies.append(e.value)
        rang = 11

        my_bomb_pre = self.observation_pre['my_bomb']
        bomb_life = copy.deepcopy(observation['bomb_life'])
        board_pre = self.observation_pre['board']
        items_pre = self.observation_pre['items']
        woods_pre = self.observation_pre['woods']
        frags_pre = self.observation_pre['frags']
        ammo_used_pre = self.observation_pre['ammo_used']
        position_pre = self.observation_pre['position']
        imove_counts_pre = self.observation_pre['imove_counts']
        reach_goals_pre = self.observation_pre['reach_goals']

        position = observation['position']
        strength = observation['blast_strength'] - 1
        alives = observation['alive']

        # 加入 idx
        observation['idx'] = self.observation_pre['idx']

        # 加入 is_alive
        observation['is_dead'] = observation['idx'] not in alives

        # 增加 ammo_used
        cur_ammo_used = self.observation_pre['ammo'] - observation['ammo']
        if cur_ammo_used > 0:
            ammo_used_pre += cur_ammo_used
        observation['ammo_used'] = ammo_used_pre

        # 加入 my_bomb
        # 首先将之前的 bomb_life -1
        my_bomb = []
        for bf_idx in range(len(my_bomb_pre)):
            my_bomb_pre[bf_idx][2] -= 1
            if my_bomb_pre[bf_idx][2] >= -2:
                my_bomb.append(my_bomb_pre[bf_idx])

        # 再加入最新放置的 bomb
        if bomb_life[position] == 9:
            my_bomb.append([position[0], position[1], 9, strength])

        observation['my_bomb'] = my_bomb

        # 加入 items
        if board_pre[position] in [extra_bomb, incr_range, kick]:
            items_pre += 1

        observation['items'] = items_pre

        # 加入 woods
        used_woods = []
        for mb in observation['my_bomb']:
            if mb[2] <= 0:  # 说明本帧刚刚爆炸, 只有刚刚爆炸才能炸到 woods
                # up, down, left, right
                for act_toward in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    mb_pos = [mb[0], mb[1]]
                    mb_str = mb[3]
                    # 在爆炸范围内深度优先
                    for t in range(mb_str):
                        mb_pos[0] = act_toward[0] + mb_pos[0]
                        mb_pos[1] = act_toward[1] + mb_pos[1]
                        # 超出界面
                        if mb_pos[0] < 0 or mb_pos[0] >= rang or mb_pos[1] < 0 or mb_pos[1] >= rang:
                            break
                        # 初次炸到木头
                        if board_pre[mb_pos[0]][mb_pos[1]] == wood and mb_pos not in used_woods:
                            woods_pre += 1
                            used_woods.append(mb_pos)

        observation['woods'] = woods_pre

        # 加入 frags
        frags = 0
        for e in enemies:
            if e not in alives:
                frags += 1
        observation['frags'] = frags

        # 总共获得了多少个 items
        self.get_items = observation['items']

        # 是否达成目标
        self.achive = position in self.observation_pre['goal_positions']

        self.interval += 1
        # 如果达成目标则重新设置目标
        if self.achive:  # or (self.interval + 1) % max_interval == 0:
            if self.achive:
                reach_goals_pre += 1
            self.interval = 0
            self.generate_item(position)

        observation['reach_goals'] = reach_goals_pre

        # 添加goal_positions
        goal_board = observation['board']
        goal_positons = []
        for x in range(len(goal_board)):
            for y in range(len(goal_board)):
                if goal_board[(x, y)] in [constants.Item.ExtraBomb.value, constants.Item.IncrRange.value,
                                          constants.Item.Kick.value]:
                    goal_positons.append((x, y))
        observation['goal_positions'] = goal_positons

        # 是否移动
        if position == position_pre:
            imove_counts_pre += 1
        observation['imove_counts'] = imove_counts_pre

        self.observation_pre = copy.deepcopy(observation)
        observations[self.train_idx] = copy.deepcopy(observation)
        self.observations = copy.deepcopy(observations)
        return observations

    def step(self, actions):
        self._intended_actions = actions

        max_blast_strength = self._agent_view_size or 10
        result = self.model.step(
            actions,
            self._board,
            self._agents,
            self._bombs,
            self._items,
            self._flames,
            max_blast_strength=max_blast_strength)

        self._board, self._agents, self._bombs, self._items, self._flames = \
            result[:5]
        obs = self.get_observations()
        done = self._get_done(obs[self.train_idx]['is_dead'])
        reward = self.get_rewards_maze_v1(done, obs[self.train_idx]['is_dead'])
        info = self._get_info(done, reward, obs[self.train_idx]['is_dead'])

        if done:
            # Callback to let the agents know that the game has ended.
            for agent in self._agents:
                agent.episode_end(reward[agent.agent_id])

        self._step_count += 1
        return obs, reward, done, info

    def reset(self, train_idx=0, goal=None):
        assert (self._agents is not None)
        self.train_idx = train_idx
        if self._init_game_state is not None:
            self.set_json_info()
        else:
            self._step_count = 0
            self.make_board()
            self.make_items()
            self._bombs = []
            self._flames = []
            self._powerups = []
            for agent_id, agent in enumerate(self._agents):
                pos = np.where(self._board == utility.agent_value(agent_id))
                row = pos[0][0]
                col = pos[1][0]
                agent.set_start_position((row, col))
                agent.reset()

        self.observation_pre = None
        self.achive = False
        self.goal = self.get_goal(goal)
        self.get_items = 0
        self.interval = 0
        return self.get_observations()

    def make_board(self):
        self._board = env_utils.make_board(self._board_size, self._num_rigid,
                                           self._num_wood, len(self._agents))
        self.generate_item((1, 1))

    def generate_item(self, position):
        for i in range(1):
            self._board = env_utils.generate_item(self._board, position, self._board_size)
            self.achive = False

    def make_items(self):
        self._items = env_utils.make_items(self._board, self._num_items)

    def get_goal(self, goal=None):
        if goal:
            return np.array(goal)

        goal = np.zeros(meas_size)

        # 7dim: [woods↑, items↑, ammo_used↑, frags↑, is_dead↑, reach_goals↑, imove_counts↑]
        goal[0] = random.uniform(0, 1)
        goal[1] = random.uniform(0, 1)
        goal[2] = random.uniform(-1, 1)
        goal[3] = 0  # frags
        goal[4] = random.uniform(-1, 0)
        goal[5] = random.uniform(0, 1)
        goal[6] = random.uniform(-0.02, 0)
        return np.array(goal)

    def _get_done(self, is_dead):
        if self._step_count >= self._max_steps:
            return True
        elif is_dead:
            return True
        return False

    def get_rewards_maze_v1(self, done, is_dead):
        if done:
            return [self.get_items, 0, 0, 0]
        return [0, 0, 0, 0]

    def _get_info(self, done, rewards, is_dead):
        if done:
            if self._step_count >= self._max_steps:
                return {
                    'result': constants.Result.Win,
                    'get items number': self.get_items,
                }
            elif is_dead:
                return {
                    'result': constants.Result.Loss,
                    'get items number': self.get_items,
                }
        return {
            'result': constants.Result.Incomplete,
        }

    @staticmethod
    def featurize(obs):
        ret = super().featurize(obs)
        message = obs['message']
        message = utility.make_np_float(message)
        return np.concatenate((ret, message))

    def get_json_info(self):
        ret = super().get_json_info()
        ret['radio_vocab_size'] = json.dumps(
            self._radio_vocab_size, cls=json_encoder)
        ret['radio_num_words'] = json.dumps(
            self._radio_num_words, cls=json_encoder)
        ret['_radio_from_agent'] = json.dumps(
            self._radio_from_agent, cls=json_encoder)
        return ret

    def set_json_info(self):
        super().set_json_info()
        self.radio_vocab_size = json.loads(
            self._init_game_state['radio_vocab_size'])
        self.radio_num_words = json.loads(
            self._init_game_state['radio_num_words'])
        self._radio_from_agent = json.loads(
            self._init_game_state['_radio_from_agent'])
