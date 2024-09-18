from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mat.envs.starcraft2.multiagentenv import MultiAgentEnv

import atexit
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
from absl import logging

import random
import gym
from gym.spaces import Discrete, Box

from typing import Any, Dict, List

import numpy as np

from grutopia.core.config import SimulatorConfig
from grutopia.core.util import log
from grutopia.core.env import BaseEnv


class GRUtopia_MAT_Env(MultiAgentEnv):

    def __init__(self,  config: SimulatorConfig, headless: bool = True, webrtc: bool = False, native: bool = False, name = 0) -> None:
        self.GRUtopia_base_env: BaseEnv = BaseEnv(config,headless,webrtc,native)
        self.name = name
        self.num_agents = len(config.config_dict['tasks'][0]['robots'])
        self.num_envs = config.config_dict['tasks'][0]['env_num']
        self.offsets = [config.config_dict['tasks'][0]['offset_size']*i for i in range(self.num_envs)]
        self.action_space = [Box(0,1,(3,))]

        # scale the observation sapce based on the number of agents
        self.observation_space = [gym.spaces.Dict({
            'cam': Box(0,1,(240,320,4)),
            'pos_ori': Box(-np.inf,np.inf,(7,)),
        })]
        self.share_observation_space = [gym.spaces.Dict({
            'cam': Box(0,1,(240,320,4)),
            'pos_ori': Box(-np.inf,np.inf,(7,)),
        })]
        self.sim_config = config
    
    def reset(self):
        local_state = self.GRUtopia_base_env.reset()[0]
        local_state = self._convert_obs_to_array(local_state)
        global_state = local_state
        return local_state, global_state
        

    def step(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        actions = self._convert_actions(actions)
        local_state = self.GRUtopia_base_env.step(actions)
        local_state = self._convert_obs_to_array(local_state)
        global_state = local_state

        reward = np.zeros((self.num_envs,self.num_agents,1))
        dones = np.zeros((self.num_envs,self.num_agents))
        infos = {}

        return local_state, global_state, reward, dones, infos
    

    def seed(self, seed):
        pass

    def _convert_actions(self,actions):
        total_actions = []

        for i in range(self.num_envs):
            total_actions.append({r['name']:{'move_by_speed':actions[i][j]} for j,r in enumerate(self.sim_config.config_dict['tasks'][0]['robots'])})

        return total_actions

        

    def _convert_obs_to_array(self,obs):
        total_array_pos_ori = []
        total_array_cam = []
        for world in obs:
            world_array_pos_ori = []
            world_array_cam = []
            for robot in obs[world]:

                robot_obs = obs[world][robot]

                position = np.concatenate((robot_obs['position'],robot_obs['orientation']))
                camera = robot_obs['camera']['rgba']/255

                world_array_pos_ori.append(position)
                world_array_cam.append(camera)

            total_array_pos_ori.append(world_array_pos_ori)
            total_array_cam.append(world_array_cam)

        return {
            'pos_ori': np.array(total_array_pos_ori),
            'cam': np.array(total_array_cam)
        }