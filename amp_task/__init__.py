# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from .AMP_env import AMP_Env
##
# Register Gym environments.
##

gym.register(
    id="Isaac-G1-AMP-Walk-v0",
    entry_point=AMP_Env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:HumanoidEnvCfg",
    },
)

gym.register(
    id="Isaac-G1-AMP-Dance-v0",
    entry_point=AMP_Env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1_DanceEnvCfg",
    },
)

gym.register(
    id="Isaac-G1-AMP-Loco-Walk-v0",
    entry_point=AMP_Env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_loco_env_cfg:G1_AMP_Loco_EnvCfg",
    },
)