import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv

from .manager.amp_motion_manager import AMPMotionManager

class AMP_Env(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        self.cfg = cfg
        super().__init__(cfg, render_mode, **kwargs)
        self.amp_motion_manager = AMPMotionManager(cfg, self, device=self.device)
        
    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        return self.amp_motion_manager.collect_reference_motions(num_samples, current_times)
    
    def step(self, action: torch.Tensor):
        returns = super().step(action)
        returns = self.amp_motion_manager.amp_step(returns)
        return returns