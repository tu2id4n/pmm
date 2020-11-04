# maze_v1
max_setps = 800
max_dijk = 9

# env_utils
num_rigid = 16
num_wood = 40

# _subproc_vec_env
# 6dim: [woods↑, items↑, ammo_used↑, frags↑, is_dead↑,  imove_counts↑]
train_goal = [1, 1, -0.1, 0.7, -2,  -0.1]
meas_size = 6
time_span = [1, 2, 4, 8, 16, 32, 64]
train_idx = 0
teammates = [train_idx, (train_idx + 2) % 4]
teammates.sort()
enemies = [(train_idx + 1) % 4, (train_idx + 3) % 4]
enemies.sort()
random_explore = True
update_eps = 0.2

# replay_buffer
hindsight = False
her_size = 15
her_K = 4
woods = 0
items = 1
ammo_used = 2
frags = 3
is_dead = 4
reach_goals = 0
imove_counts = 1

# dfp_policy
import tensorflow as tf
from stable_baselines.a2c.utils import ortho_init

pgn = False
init_scale = 1
# conv_init = ortho_init(init_scale)
conv_init = tf.glorot_normal_initializer()
linear_init = tf.glorot_normal_initializer()

# dfp
buffer_size = 20000
learning_starts = 1000
exploration_final_eps = 0.2
exploration_fraction = 0.05
n_actions = 122
gamma = 0.99
batch_size = 32
lr_decay_step = 5e5
decay_rate = 0.3
