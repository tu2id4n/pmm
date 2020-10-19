import argparse

def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 通用参数
    parser.add_argument('--model_type', help="DFP", default='dfp')
    parser.add_argument('--policy_type', help="DFPPolicy", default='dfppolicy')

    # train
    parser.add_argument('--env', help='环境名称', type=str, default='PommeRadioCompetition-v21')
    parser.add_argument('--num_env', help='并行环境数量，默认根据 CPU 来选取数量', default=None, type=int)
    parser.add_argument('--num_timesteps', help='强化学习算法训练所用步数', type=float, default=1e6)
    parser.add_argument('--nsteps', type=int, default=128)
    parser.add_argument('--save_interval', help='保存间隔', type=float, default=1e5)

    # save and load
    parser.add_argument('--save_path', help='保存模型', default=None, type=str)
    parser.add_argument('--load_path', help='加载模型', default=None, type=str)
    parser.add_argument('--log_path', help='日志', default=None, type=str)
    parser.add_argument('--hindsight', help='日志', default=False, type=bool)

    return parser
