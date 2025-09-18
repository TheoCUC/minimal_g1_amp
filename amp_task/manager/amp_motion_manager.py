import os
import torch
import numpy as np
import gymnasium as gym

from isaaclab.managers.manager_base import ManagerBase, ManagerTermBase

from ..motions import MotionLoader
from ..amp_mdp.utils import compute_obs

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "motions")

class AMPMotionManager:
    """Motion imitation manager for AMP training."""

    def __init__(self, cfg, env, device):
        # super().__init__(cfg=cfg, env=env)
        self.cfg = cfg

        self.amp_observation_size = self.cfg.amp.num_amp_observations * self.cfg.amp.num_amp_observation_space
        env.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (env.scene.num_envs, self.cfg.amp.num_amp_observations, self.cfg.amp.num_amp_observation_space), device=device
        )
        
        motion_file = self.cfg.amp.motion_file
        self._motion_loader = MotionLoader(motion_file=motion_file, device=device)
        
        self.ref_body_index = env.scene['robot'].data.body_names.index(self.cfg.amp.reference_body)
        self.key_body_indexes = [env.scene['robot'].data.body_names.index(name) for name in self.cfg.amp.key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(env.scene['robot'].data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.amp.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(self.cfg.amp.key_body_names)

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.amp.num_amp_observations)
        ).flatten()
        # get motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        # compute AMP observation

        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )

        return amp_observation.view(-1, self.amp_observation_size)
    
    def amp_step(self, env_returns):
        # update AMP observation history
        for i in reversed(range(self.cfg.amp.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation
        self.amp_observation_buffer[:, 0] = env_returns[0]['amp_obs'].clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}
        env_returns[-1]["amp_obs"] = self.amp_observation_buffer.view(-1, self.amp_observation_size)
        return env_returns