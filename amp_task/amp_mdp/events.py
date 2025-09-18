
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def reset_root_state_amp(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    num_samples = env_ids.shape[0]
    motion_loader = env.amp_motion_manager._motion_loader

    times = motion_loader.sample_times(num_samples)
    
    # sample random motions
    (
        dof_positions,
        dof_velocities,
        body_positions,
        body_rotations,
        body_linear_velocities,
        body_angular_velocities,
    ) = motion_loader.sample(num_samples=num_samples, times=times)
    
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    
    motion_dof_indexes = motion_loader.get_dof_index(asset.data.joint_names)
    
    motion_torso_index = motion_loader.get_body_index([env.cfg.amp.reference_body])[0]

    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, 0:3] = body_positions[:, motion_torso_index] + env.scene.env_origins[env_ids]
    root_state[:, 2] += 0.05  # lift the humanoid slightly to avoid collisions with the ground
    root_state[:, 3:7] = body_rotations[:, motion_torso_index]
    root_state[:, 7:10] = body_linear_velocities[:, motion_torso_index]
    root_state[:, 10:13] = body_angular_velocities[:, motion_torso_index]
    # get DOFs state
    dof_pos = dof_positions[:, motion_dof_indexes]
    dof_vel = dof_velocities[:, motion_dof_indexes]

    # update AMP observation
    amp_observations = env.amp_motion_manager.collect_reference_motions(num_samples, times)
    env.amp_motion_manager.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, env.cfg.amp.num_amp_observations, -1)
    
    asset.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
    asset.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
    asset.write_joint_state_to_sim(dof_pos, dof_vel, None, env_ids)