# replay_buffer
hindsight = True
her_size = 15
her_K = 4
# 7dim: [woods↑, items↑, ammo_used↑, frags↑, is_dead↑, reach_goals↑, imove_counts↑]
woods = 0
items = 1
ammo_used = 2
frags = 3
is_dead = 4
reach_goals = 0
imove_counts = 1

# _subproc_vec_env
# 7dim: [woods↑, items↑, ammo_used↑, frags↑, is_dead↑, reach_goals↑, imove_counts↑]
time_span = [1, 8, 16, 32]
train_goal = [5, -1]
train_idx = 0
teammates = [train_idx, (train_idx + 2) % 4]
teammates.sort()
enemies = [(train_idx + 1) % 4, (train_idx + 3) % 4]
enemies.sort()
random_explore = False
update_eps = 1

# dfp
buffer_size = 20000
learning_starts = 10000
exploration_final_eps = 0.2
exploration_fraction = 0.05
n_actions = 4
gamma = 0.9
batch_size = 32
lr_decay_step = 1e6
decay_rate = 0.3

# dfp_policy
import tensorflow as tf
from stable_baselines.a2c.utils import ortho_init

pgn = False
init_scale = 1
conv_init = ortho_init(init_scale)
linear_init = tf.glorot_normal_initializer()

# maze_v1
max_setps = 800
meas_size = 1
max_interval = 100

# env_utils
num_rigid = 0
