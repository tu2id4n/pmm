""" maze v1 """
from gym import spaces
import numpy as np

from pommerman import constants
from pommerman import utility
from pommerman.envs import v0
import copy
import random
from . import env_utils
from _common import _constants
from _common import featurize

max_setps = _constants.max_setps
meas_size = _constants.meas_size
max_dijk = _constants.max_dijk


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

    def get_reset_observations(self, ):
        observations = super().get_observations()  # 已经获得新的self.observations了

        for obs in observations:
            obs['message'] = self._radio_from_agent[obs['teammate']]

        observation = observations[self.train_idx]
        observation['my_bomb'] = []
        observation['woods'] = 0
        observation['frags'] = 0
        observation['items'] = 0
        observation['idx'] = observation['board'][observation['position']]
        observation['ammo_used'] = 0
        observation['goal'] = self.goal
        observation['imove_counts'] = 0
        observation['is_dead'] = False
        observation['reach'] = 0

        self.observation_pre = copy.deepcopy(observation)
        observations[self.train_idx] = copy.deepcopy(observation)
        self.observations = copy.deepcopy(observations)

        return self.observations

    def get_observations(self, ):
        observations = super().get_observations()  # 已经获得新的self.observations了
        for i in range(len(observations)):
            observations[i]['step_count'] = self._step_count
        for obs in observations:
            obs['message'] = self._radio_from_agent[obs['teammate']]

        observation = observations[self.train_idx]

        if observation['step_count'] == self.observation_pre['step_count']:
            self.observations[self.train_idx] = self.observation_pre
            return self.observations

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
        reach_pre = self.observation_pre['reach']

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
        my_bomb = []  # [x, y, -2 ~ 9, strength]
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

        # 加入 woods, frags
        used_woods = []
        # [x, y, -2 ~ 9, strength]
        for mb in observation['my_bomb']:
            if mb[2] == 9:  # 说明本帧刚刚放置, 炸到 woods
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
                        if board_pre[mb_pos[0]][mb_pos[1]] in enemies:
                            frags_pre += 1

        observation['woods'] = woods_pre
        observation['frags'] = frags_pre

        # 总共获得了多少个 items
        self.get_items = observation['items']

        dijk_pos = featurize.extra_position(self.dijk_act, observation)
        self.is_dijk = dijk_pos == position

        if self._intended_actions[0] == 0:  # 没有移动 即为 imove
            imove_counts_pre += 1
        observation['imove_counts'] = imove_counts_pre
        if self.is_dijk and self._intended_actions[0] != 5:
            reach_pre += 1
        observation['reach'] = reach_pre

        self.observation_pre = copy.deepcopy(observation)
        observations[self.train_idx] = copy.deepcopy(observation)
        self.observations = copy.deepcopy(observations)

        return self.observations

    def step(self, actions):
        self._step_count += 1

        if self.is_dijk or self.dijk_step > max_dijk:  # 重启 dijksta
            self.dijk_step = 0
            self.dijk_act = actions[0]
            self.is_dijk = False

        self.dijk_step += 1
        actions[0] = featurize.dijkstra_act(self.observation_pre, self.dijk_act)
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
        done = self._get_done()
        reward = self.get_rewards(done)
        info = self._get_info(done, reward)

        if done:
            # Callback to let the agents know that the game has ended.
            for agent in self._agents:
                agent.episode_end(reward[agent.agent_id])

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
        # self.achive = False
        self.goal = self.get_goal(goal)
        self.get_items = 0
        self.is_dijk = True
        self.dijk_act = 0
        self.dijk_step = 0

        return self.get_reset_observations()

    # 7dim: [woods↑, items↑, ammo_used↑, frags↑, is_dead↑, reach_goals↑, imove_counts↑]
    def get_goal(self, goal=None):
        if goal:
            return np.array(goal)

        goal = np.zeros(meas_size)
        for i in range(meas_size):
            goal[i] = random.uniform(-1, 1)

        return np.array(goal)

    def _get_done(self):
        alive = [agent for agent in self._agents if agent.is_alive]
        alive_ids = sorted([agent.agent_id for agent in alive])

        if self._step_count >= self._max_steps:
            return True
        elif self.train_idx is not None and self.train_idx not in alive_ids:
            return True
        elif any([
            len(alive_ids) <= 1,
            alive_ids == [0, 2],
            alive_ids == [1, 3],
        ]):
            return True
        return False

    def _get_info(self, done, rewards):
        alive = [agent for agent in self._agents if agent.is_alive]
        alive_ids = sorted([agent.agent_id for agent in alive])
        if done:
            if self._step_count >= self._max_steps:
                return {
                    'result': constants.Result.Tie,
                }
            elif any([
                alive_ids == [0, 2],
                alive_ids == [0],
                alive_ids == [2]
            ]):

                return {
                    'result': constants.Result.Win,
                    'winners': [0, 2],
                }

            else:
                return {
                    'result': constants.Result.Loss,
                    'winners': [1, 3],
                }

        return {
            'result': constants.Result.Incomplete,
        }

    def get_rewards(self, done):

        alive = [agent for agent in self._agents if agent.is_alive]
        alive_ids = sorted([agent.agent_id for agent in alive])
        if done:
            if self._step_count >= self._max_steps:
                return [-1, -1, -1, -1]
            elif any([
                alive_ids == [0, 2],
                alive_ids == [0],
                alive_ids == [2]
            ]):

                return [1, -1, 1, -1]

            else:
                return [-1, 1, -1, 1]

        return [0, 0, 0, 0]

    def make_board(self):
        self._board = utility.make_board(self._board_size, _constants.num_rigid,
                                         _constants.num_wood, len(self._agents))

    def make_items(self):
        self._items = utility.make_items(self._board, _constants.num_item)

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
