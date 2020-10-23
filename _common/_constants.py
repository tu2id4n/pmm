# replay_buffer
hindsight = False
her_pb = 0.5
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
train_goal = [1, -0.01]
train_idx = 0
teammates = [train_idx, (train_idx + 2) % 4]
teammates.sort()
enemies = [(train_idx + 1) % 4, (train_idx + 3) % 4]
enemies.sort()
random_explore = False

# dfp
buffer_size = 20000
learning_starts = 10000

time_span = [1, 2, 4, 8, 16, 32]
exploration_final_eps = 0.2
exploration_fraction = 0.05
update_eps = 1
n_actions = 4
gamma = 0.99

batch_size = 128
lr_decay_step = 1e6
decay_rate = 0.3

# dfp_policy
import tensorflow as tf
from stable_baselines.a2c.utils import ortho_init

pgn = False
conv_init = ortho_init(init_scale=1)
linear_init = tf.glorot_normal_initializer()


