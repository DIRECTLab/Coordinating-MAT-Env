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

        self.step_count = 0
        self.nextcommand = 0

        self.commands = [[0, 0,0] for i in range(self.num_envs)]

        # scale the observation sapce based on the number of agents
        self.observation_space = [gym.spaces.Dict({
            'cam': Box(0,1,(240,320,4)),
            'pos_ori': Box(-np.inf,np.inf,(10,))
        })]
        self.share_observation_space = [gym.spaces.Dict({
            'cam': Box(0,1,(240,320,4)),
            'pos_ori': Box(-np.inf,np.inf,(10,))
        })]

        self.sim_config = config

        # Define the dimensions and boundaries of the rectangle at the center of the offset area
        self.rectangle_width = 10  # Adjust as needed
        self.rectangle_height = 10  # Adjust as needed
        self.rectangle_x_min = 0
        self.rectangle_x_max = self.rectangle_width
        self.rectangle_y_min = 0
        self.rectangle_y_max = self.rectangle_height
        # Define grid resilution size for coverage tracking
        self.grid_size_x = 100  # Adjust as needed
        self.grid_size_y = 100  # Adjust as needed

        # Define maximum steps per episode
        self.max_steps = 1000  # Adjust as needed

        # Initialize the coverage grid
        self.coverage_grid = np.zeros((self.num_envs, self.grid_size_x, self.grid_size_y))

        # Initialize step counters
        self.step_count = 0
        self.nextcommand = 0
    
    def _generate_new_commands(self):
        return [[self.rng.random()*20+self.offsets[i], self.rng.random()*20,1.05] for i in range(self.num_envs)]

    def reset(self):
    # Reset coverage grid
        self.coverage_grid = np.zeros((self.num_envs, self.grid_size_x, self.grid_size_y))

        local_state = self.GRUtopia_base_env.reset()[0]
        local_state = self._convert_obs_to_array(local_state)
        global_state = local_state
        self.commands = self._generate_new_commands()
        self.step_count=0
        self.nextcommand = 0
        return local_state, global_state
        

    def step(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        actions = self._convert_actions(actions)
        local_state = self.GRUtopia_base_env.step(actions)
        local_state = self._convert_obs_to_array(local_state)
        global_state = local_state

        # Initialize reward
        reward = np.zeros((self.num_envs, self.num_agents, 1))

        # Update coverage grid and compute reward
        for i in range(self.num_envs):
            # Define per-environment rectangle boundaries based on offset
            rectangle_x_min = self.offsets[i]
            rectangle_x_max = rectangle_x_min + self.rectangle_width
            rectangle_y_min = 0  # Assuming no offset in y-direction; adjust if needed
            rectangle_y_max = rectangle_y_min + self.rectangle_height

            for j in range(self.num_agents):
                # Get the agent's current position
                pos = local_state['pos_ori'][i][j][:2]
                x_pos, y_pos = pos[0], pos[1]

                # Check if the agent is within the rectangle boundaries
                out_of_bounds = False
                if not (rectangle_x_min <= x_pos <= rectangle_x_max) or \
                not (rectangle_y_min <= y_pos <= rectangle_y_max):
                    # Agent is out of bounds
                    # sacle the reward based on the distance from the center of the rectangle
                    reward[i, j, 0] -= 1 * np.linalg.norm([x_pos-self.rectangle_x_max/2,y_pos-self.rectangle_y_max/2])
                    out_of_bounds = True

                # Proceed only if the agent is within bounds
                if not out_of_bounds:
                    # Map position to grid coordinates relative to the rectangle boundaries
                    x_rel = x_pos - rectangle_x_min
                    y_rel = y_pos - rectangle_y_min

                    x = int(x_rel / (rectangle_x_max - rectangle_x_min) * self.grid_size_x)
                    y = int(y_rel / (rectangle_y_max - rectangle_y_min) * self.grid_size_y)

                    # Ensure the indices are within the grid bounds
                    x = np.clip(x, 0, self.grid_size_x - 1)
                    y = np.clip(y, 0, self.grid_size_y - 1)

                    if self.coverage_grid[i, x, y] == 0:
                        # Agent has visited a new cell
                        reward[i, j, 0] += 1  # Positive reward for new cell
                        self.coverage_grid[i, x, y] = 1
                    else:
                        # Agent is revisiting a cell
                        reward[i, j, 0] += 0  # No reward or small penalty

                # Time penalty to encourage faster completion
                reward[i, j, 0] -= 0.01  # Small negative reward per time step

        # Check if the episode should end
        if self.step_count >= self.max_steps or np.all(self.coverage_grid == 1):
            # Episode ends either due to max steps or full coverage
            dones = np.ones((self.num_envs, self.num_agents))

            # Compute total coverage per environment
            total_coverage = np.sum(self.coverage_grid, axis=(1, 2)) / (self.grid_size_x * self.grid_size_y)
            total_coverage = total_coverage.reshape(self.num_envs, 1, 1)

            # Scale the final reward inversely with time taken to encourage quicker completion
            time_bonus = (self.max_steps - self.step_count) / self.max_steps
            final_reward = total_coverage * time_bonus * 10  # Adjust scaling factor as needed
            reward += final_reward  # Add the final reward to existing rewards

            infos = {}

            # Reset the environment
            local_state, global_state = self.reset()
            return local_state, global_state, reward, dones, infos

        # Update counters
        self.step_count += 1
        self.nextcommand += 1

        dones = np.zeros((self.num_envs, self.num_agents))
        infos = {'commands': self.commands}

        return local_state, global_state, reward, dones, infos

    

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def _convert_actions(self,actions):
        total_actions = []

        for i in range(self.num_envs):
            total_actions.append({r['name']:{'move_by_speed':actions[i][j]} for j,r in enumerate(self.sim_config.config_dict['tasks'][0]['robots'])})

        return total_actions

    def _convert_obs_to_array(self,obs):
        total_array_pos_ori = []
        total_array_cam = []
        for i,world in enumerate(obs):
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

        commands = np.array(self.commands)[:, np.newaxis, :]
        commands = np.tile(commands, (1, self.num_agents, 1))



        return {
            'pos_ori': np.concatenate((commands,np.array(total_array_pos_ori)),axis=-1),
            'cam': np.array(total_array_cam)
        }