# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject

from .utils import quaternion_to_tangent_and_normal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def body_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Computes the positions of the body in world coordinates.
    Returns:
        torch.Tensor: A tensor of shape (N, 3) where N is the batch
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ref_body_index = asset.data.body_names.index(env.cfg.amp.reference_body)
    return asset.data.body_pos_w[:, ref_body_index][:, 2:3]

def body_quat_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Computes the orientation of the body in world coordinates and converts it to tangent and normal representation.
    Returns:
        torch.Tensor: A tensor of shape (N, 6) where N is the batch
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ref_body_index = asset.data.body_names.index(env.cfg.amp.reference_body)
    return quaternion_to_tangent_and_normal(asset.data.body_quat_w[:, ref_body_index])

def body_lin_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Computes the linear velocities of the body in world coordinates.
    Returns:
        torch.Tensor: A tensor of shape (N, 3) where N is the batch
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ref_body_index = asset.data.body_names.index(env.cfg.amp.reference_body)
    return asset.data.body_lin_vel_w[:, ref_body_index]

def body_ang_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Computes the angular velocities of the body in world coordinates.
    Returns:
        torch.Tensor: A tensor of shape (N, 3) where N is the batch
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ref_body_index = asset.data.body_names.index(env.cfg.amp.reference_body)
    return asset.data.body_ang_vel_w[:, ref_body_index]

def key_body_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Computes the relative positions of key bodies with respect to a reference body in world coordinates.
    Args:
        env (ManagerBasedEnv): The environment containing the scene and configuration.
        asset_cfg (SceneEntityCfg, optional): Configuration for the asset (e.g., robot) to retrieve data from. Defaults to SceneEntityCfg("robot").
    Returns:
        torch.Tensor: A tensor of shape (N, M) where N is the batch size and M is the flattened dimension of the relative positions of key bodies.
    """
    
    asset: RigidObject = env.scene[asset_cfg.name]
    key_body_indexes = [asset.data.body_names.index(name) for name in env.cfg.amp.key_body_names]
    key_body_positions = asset.data.body_pos_w[:, key_body_indexes]
    root_positions = asset.data.body_pos_w[:, asset.data.body_names.index(env.cfg.amp.reference_body)].unsqueeze(-2)
    return (key_body_positions - root_positions).view(key_body_positions.shape[0], -1)

