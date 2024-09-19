#!/usr/bin/env python
import multiprocessing
import sys
import os
import socket
import numpy as np
from pathlib import Path
import torch
import setproctitle

from mat.config import get_config
from mat.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from src.runner.grutopia_multi_runner import GRUtopiaRunner as Runner
from src.envs.grutopia_mat_env import GRUtopia_MAT_Env

from grutopia.core.config import SimulatorConfig

file_path = './GRUtopia/src/configs/train_grutopia_mat.yaml'
sim_config = SimulatorConfig(file_path)

SEED = 0
HEADLESS = True

def make_train_env(empty=False):

    env = GRUtopia_MAT_Env(sim_config, HEADLESS,empty=empty)
    env.seed(0)

    return env

def make_eval_env(all_args):

    def get_env_fn(rank):
        def init_env():
            env = GRUtopia_MAT_Env(sim_config, HEADLESS)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env

    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_false', default=True)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_false', default=True)

    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):

    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mat" or all_args.algorithm_name == "mat_dec":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True

    all_args.env_name = sim_config.config_dict['tasks'][0]['name']

    print("choose to use gpu...")
    device = torch.device("cuda:0")
    torch.set_num_threads(all_args.n_training_threads)
    if all_args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    num_agents = len(sim_config.config_dict['tasks'][0]['robots'])
    all_args.run_dir = run_dir
    all_args.num_env_steps = 100000
    all_args.episode_length = 300
    all_args.n_rollout_threads = sim_config.config_dict['tasks'][0]['env_num']
    
    envs_fn = make_train_env
    eval_envs = None

    config = {
        "all_args": all_args,
        "envs_fn": envs_fn,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
        "sim_config": sim_config
    }

    runner = Runner(config)
    runner.run()

    # post process
    # envs.close()
    # if all_args.use_eval and eval_envs is not envs:
    #     eval_envs.close()
        # for eval_env in eval_envs:
        #     eval_env.close()

    # runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    # runner.writter.close()


if __name__ == "__main__":
     multiprocessing.set_start_method('spawn', force=True)
     main(sys.argv[1:])
