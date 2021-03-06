import sys
import os
import multiprocessing
import tensorflow as tf
from _common import _cmd_utils, _subproc_vec_env, run_utils
from _baselines import DFP
from _policies import DFPPolicy


def dfp_train():
    print(args)
    # Mutiprocessing
    # config = tf.ConfigProto()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # config.gpu_options.allow_growth = True
    # num_envs = args.num_env or multiprocessing.cpu_count()

    total_timesteps = int(args.num_timesteps)

    num_envs = 1
    envs = [run_utils.make_envs(args.env) for _ in range(num_envs)]

    env = _subproc_vec_env.SubprocVecEnv(envs)
    if args.load_path is None:
        model = DFP(policy=DFPPolicy, env=env, tensorboard_log=args.log_path)
    else:
        model = DFP.load(load_path=args.load_path, tensorboard_log=args.log_path, env=env)

    model.learn(total_timesteps=total_timesteps, save_path=args.save_path, save_interval=args.save_interval)

    env.close()


if __name__ == '__main__':
    arg_parser = _cmd_utils.arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)

    dfp_train()
