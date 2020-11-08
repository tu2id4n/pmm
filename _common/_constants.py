# maze_v1
max_setps = 800
max_dijk = 9

# env_utils
num_rigid = 36
num_wood = 36
num_item = 20

# _subproc_vec_env
# 7dim: [woods↑, items↑, ammo_used↑, frags↑, is_dead↑,  imove_counts↑, reach↑]
train_goal = [1, 1, -0.1, 0.2, -3, -0.1, 0.2]
meas_size = 7
time_span = [1, 2, 4, 8, 16, 32]
train_idx = 0
teammates = [train_idx, (train_idx + 2) % 4]
teammates.sort()
enemies = [(train_idx + 1) % 4, (train_idx + 3) % 4]
enemies.sort()
random_explore = True
update_eps = 0.2
pgn = False
n_actions = 122

# replay_buffer
hindsight = False
her_size = 30
her_K = 4

# dfp_policy
import tensorflow as tf
from stable_baselines.a2c.utils import ortho_init

init_scale = 1
conv_init = ortho_init(init_scale)
# conv_init = tf.glorot_normal_initializer()
linear_init = tf.glorot_normal_initializer()

# dfp
buffer_size = 50000
learning_starts = 10000
exploration_final_eps = 0.2
exploration_fraction = 0.05
gamma = 0.99
batch_size = 64
lr_decay_step = 5e5
decay_rate = 0.3
