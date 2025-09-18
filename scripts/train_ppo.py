
# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory

from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from models.ppo import Shared, get_ppo_cfg

RESUME = False
# if you want to resume training, set the path to the checkpoint file
# RESUME = "xxxx/checkpoints/agent_80000.pt"

set_seed(42)  # e.g. `set_seed(42)` for fixed seed

# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Humanoid-28-v0", cli_args=["--headless"],)
env = wrap_env(env)

device = env.device

cfg = get_ppo_cfg(env=env, device=device)
# if you want to log the experiment to wandb
# cfg = get_ppo_cfg(env=env, device=device, use_wandb=True, wandb_project="my-project")
# if you want to resume training and log the experiment to wandb
# cfg = get_ppo_cfg(env=env, device=device, use_wandb=True, wandb_project="my-project", resume_id="xxxx")

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

if RESUME:
    agent.load(RESUME)
    
# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 32000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()