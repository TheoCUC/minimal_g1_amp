# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp
from .amp_mdp import observations as amp_mdp
from .amp_mdp.events import reset_root_state_amp


from .g1_cfg import G1_CFG
import os

##
# Scene definition
##

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a humanoid robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # robot
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"],
        offset={
            ".*_hip_pitch_joint": 0.1746,
            "waist_.*": 0.0,
            "left_hip_roll_joint": 1.2217,
            "right_hip_roll_joint": -1.2217,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 1.3963,
            ".*_shoulder_pitch_joint": -0.2094,
            ".*_ankle_pitch_joint": -0.1745,
            "left_shoulder_roll_joint": 0.3316,
            "right_shoulder_roll_joint": -0.3316,
            ".*_ankle_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.5236,
            ".*_wrist_roll_joint": 0.0,
            ".*_wrist_pitch_joint": 0.0,
            ".*_wrist_yaw_joint": 0.0,
        },
        scale={
            ".*_hip_pitch_joint": 4.8694,
            "waist_yaw_joint": 4.7124,
            ".*hip_roll_joint": 3.1416,
            "waist_roll_joint": 0.9360,
            ".*_hip_yaw_joint": 4.9637,
            "waist_pitch_joint": 0.9360,
            ".*_knee_joint": 2.6704,
            ".*_shoulder_pitch_joint": 5.1836,
            ".*_ankle_pitch_joint": 1.2566,
            ".*_shoulder_roll_joint": 3.4557,
            ".*_ankle_roll_joint": 0.4712,
            ".*_shoulder_yaw_joint": 4.7124,
            ".*_elbow_joint": 2.8274,
            ".*_wrist_roll_joint": 3.5500,
            ".*_wrist_pitch_joint": 2.9060,
            ".*_wrist_yaw_joint": 2.9060,
        })

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""
        dof_positions = ObsTerm(func=mdp.joint_pos)
        dof_velocities = ObsTerm(func=mdp.joint_vel)
        root_positions = ObsTerm(func=amp_mdp.body_pos_w)
        root_rotations =ObsTerm(func=amp_mdp.body_quat_w)
        root_linear_velocities =ObsTerm(func=amp_mdp.body_lin_vel_w)
        root_angular_velocities =ObsTerm(func=amp_mdp.body_ang_vel_w)
        key_body_positions =ObsTerm(func=amp_mdp.key_body_pos_w)
        actions = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AmpCfg(ObsGroup):
        """Observations for the policy."""
        dof_positions = ObsTerm(func=mdp.joint_pos)
        dof_velocities = ObsTerm(func=mdp.joint_vel)
        root_positions = ObsTerm(func=amp_mdp.body_pos_w)
        root_rotations =ObsTerm(func=amp_mdp.body_quat_w)
        root_linear_velocities =ObsTerm(func=amp_mdp.body_lin_vel_w)
        root_angular_velocities =ObsTerm(func=amp_mdp.body_ang_vel_w)
        key_body_positions =ObsTerm(func=amp_mdp.key_body_pos_w)
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    # For skrl-based AMP implementation, we need a separate observation space for the AMP, named 'amp_obs'
    amp_obs: AmpCfg = AmpCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    
    reset_amp = EventTerm(
        func=reset_root_state_amp,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})


MOTIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "motions")

@configclass
class AMPCfg:
    reference_body = "pelvis"
    num_amp_observations = 2
    num_amp_observation_space = 101
    motion_file = os.path.join(MOTIONS_DIR, "g1_walk.npz")
    key_body_names = [ 
        "left_shoulder_pitch_link",
        "right_shoulder_pitch_link",
        "left_elbow_link",
        "right_elbow_link",
        "right_hip_yaw_link",
        "left_hip_yaw_link",
        "right_rubber_hand",
        "left_rubber_hand",
        "right_ankle_roll_link",
        "left_ankle_roll_link"
    ]

@configclass
class HumanoidEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    amp: AMPCfg = AMPCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 1 / 60.0
        self.sim.render_interval = self.decimation

@configclass
class G1_DanceEnvCfg(HumanoidEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.amp.motion_file = os.path.join(MOTIONS_DIR, "g1_dance.npz")