from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .multiagentenv import MultiAgentEnv
from .smac_maps import get_map_params

import atexit
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
from absl import logging

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol

from . import feature_translation as ft

import random
from gym.spaces import Discrete

from typing import Any, Dict, List

import numpy as np

from grutopia.core.config import SimulatorConfig
from grutopia.core.util import log
from grutopia.core.env import BaseEnv


class GRUtopia_MAT_Env(MultiAgentEnv):

    def __init__(self,  config: SimulatorConfig, headless: bool = True, webrtc: bool = False, native: bool = False, name = 0) -> None:
        self.GRUtopia_base_env: BaseEnv = BaseEnv(config,headless,webrtc,native)
        self.name = name
    
    def reset(self):
        return self.GRUtopia_base_env.reset()

    def step(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.GRUtopia_base_env.step(actions)