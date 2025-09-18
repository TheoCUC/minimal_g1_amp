# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from models.ppo import Shared, get_ppo_cfg
import os
import gymnasium as gym

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed
EXPERIMENT_PATH = "runs/UnitreeG1_AMP/"
# if you want to specify a particular experiment ID, you can set it here, such as EXPERIMENT_ID = "2024-06-10_15-30-00"
EXPERIMENT_ID = None
CHECK_POINT_NAME = "best_agent.pt"

# try to load find the checkpoint path
try:
    if EXPERIMENT_ID is None:
        # Find the latest folder in EXPERIMENT_PATH
        folders = [f for f in os.listdir(EXPERIMENT_PATH) if os.path.isdir(os.path.join(EXPERIMENT_PATH, f))]
        if not folders:
            raise FileNotFoundError("No experiment folders found in {}".format(EXPERIMENT_PATH))
        latest_folder = max(folders, key=lambda x: os.path.getmtime(os.path.join(EXPERIMENT_PATH, x)))
        EXPERIMENT_ID = latest_folder
        CHECKPOINT_PATH = os.path.join(EXPERIMENT_PATH, EXPERIMENT_ID, "checkpoints", CHECK_POINT_NAME)
        
    else:
        CHECKPOINT_PATH = os.path.join(EXPERIMENT_PATH, EXPERIMENT_ID, "checkpoints", CHECK_POINT_NAME)
        
    print(f"Using checkpoint: {CHECKPOINT_PATH}")
except FileNotFoundError:
    breakpoint()

VIDEO = True
VIDEO_LENGTH = 500  # in frames
video_kwargs = {
    "video_folder": os.path.join(EXPERIMENT_PATH, "videos", "play"),
    "step_trigger": lambda step: step == 0,
    "video_length": VIDEO_LENGTH,
    "disable_logger": True,
}

# load and wrap the Isaac Lab environment
# env = load_isaaclab_env(task_name="Isaac-Humanoid-AMP-Run-Direct-v0", cli_args=["--video", "--headless", "--enable_cameras"],)
env = load_isaaclab_env(task_name="Isaac-Humanoid-28-v0", cli_args=["--video", "--headless", "--enable_cameras"],)
if VIDEO:
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
env = wrap_env(env)

device = env.device

cfg = get_ppo_cfg(env=env, device=device)
cfg["experiment"]["write_interval"] = 0  # don't log to TensorBoard
cfg["experiment"]["checkpoint_interval"] = 0  # don't save checkpoints

memory = RandomMemory(memory_size=32, num_envs=env.num_envs, device=device)

models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model

# instantiate the agent's models (function approximators).
agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

cfg_trainer = {"timesteps": VIDEO_LENGTH, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

agent.load(CHECKPOINT_PATH)
# start evaluation
trainer.eval()
