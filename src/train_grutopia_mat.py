#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

from mat.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from src.runner.grutopia_multi_runner import GRUtopiaRunner as Runner
from src.envs.grutopia_mat_env import GRUtopia_MAT_Env

from grutopia.core.config import SimulatorConfig

file_path = './GRUtopia/src/configs/train_grutopia_mat.yaml'
sim_config = SimulatorConfig(file_path)

N_ROLLOUT_THREADS = 1
SEED = 0

def make_train_env(config):

    def get_env_fn(rank):
        def init_env():
            env = GRUtopia_MAT_Env(config, rank)
            env.seed(SEED)
            return env
        return init_env

    return ShareSubprocVecEnv([get_env_fn(i) for i in range(N_ROLLOUT_THREADS)])

# def make_eval_env(all_args):
#     eval_maps = all_args.eval_maps
#     if all_args.n_eval_rollout_threads % len(eval_maps) != 0:
#         raise NotImplementedError
#     threads_per_map = all_args.n_eval_rollout_threads / len(eval_maps)

#     def get_env_fn(rank):
#         def init_env():
#             map_name = eval_maps[int(rank/threads_per_map)]
#             env = RandomStarCraft2EnvMulti(all_args, map_name)
#             env.seed(all_args.seed * 50000 + rank * 10000)
#             return env
#         return init_env

#     return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def main(args):

    device = torch.device("cuda:0")
    torch.set_num_threads(N_ROLLOUT_THREADS)

    
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    envs = make_train_env(sim_config)
    # eval_envs = make_eval_env(sim_config) if True else None

    config = {
        "config": sim_config,
        "envs": envs,
        # "eval_envs": eval_envs,
        "device": device,
    }

    runner = Runner(config)
    runner.run()

    # post process
    # envs.close()
    # if True and eval_envs is not envs:
    #     eval_envs.close()
        # for eval_env in eval_envs:
        #     eval_env.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
