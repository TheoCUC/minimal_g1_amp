# minimal_g1_amp

This repository is a minimal example of achieving humanoid walking in the [IsaacLab](https://github.com/isaac-sim/IsaacLab) environment through Manager-Based Reinforcement Learning. It is modified based on the `Humanoid-28-AMP` task in `IsaacLab`.

> Part of the code comes from [Humanoid-AMP](https://github.com/linden713/humanoid_amp)

## 🚀 Quick Start

1. Link the task to the `IsaacLab` repository, this is to quickly register the environment to `isaaclab.tasks`.

```bash
ln -s minimal_g1_amp/amp_task YourPATHOFISAACLAB/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/g1_amp
```

2. Execute training script

```bash
python scripts/train_amp.py
```

> This training script is organized according to the recommended approach in the `skrl` tutorial, but you can also structure the code following the conventions in `IsaacLab.scripts.reinforcement_learning`.

3. Record a video after completing the training.

```bash
python scripts/play_amp.py
```

Afterwards, you can find the recent experiment and videos in the `runs` directory.

4. Learning the dance task

in the `scripts/train_amp.py` and `scripts/play_amp.py` , edit

```python
# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-G1-AMP-Walk-v0", cli_args=["--headless"],)
```

to

```python
# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-G1-AMP-Dance-v0", cli_args=["--headless"],)
```

This essentially involves inheriting the Walk task from `amp_task/g1_amp_env_cfg.py`, then replacing the AMP motion file, and registering it as a new task in `amp_task/__init__.py`. You can also implement your own task following this approach.

This motion file comes from `dance1_subject1` of the `LAFAN` dataset.

## 🤖 Use your own AMP data

High-quality imitation data is crucial for the learning process of the AMP algorithm. In this repository, you can modify the `motion_file` in `AMPCfg` within `amp_task/g1_amp_env_cfg.py` to change the motion data file.

### Unitree

The retargeted dataset can be downloaded directly from HuggingFace.

- [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
- [ember-lab-berkeley/AMASS_Retargeted_for_G1](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1)

### GMR

You can also use [GMR](https://github.com/YanjieZe/GMR) to redirect and convert the original AMASS or LAFAN dataset into the `npz` dataset required by this repository.

1. First, redirect the action file to your own robot following the method described in [GMR](https://github.com/YanjieZe/GMR).

- Download the LaFan dataset
- Install [GMR](https://github.com/YanjieZe/GMR)
- Run the [GMR](https://github.com/YanjieZe/GMR) retargeting process

```bash
# single motion
python scripts/smplx_to_robot.py --smplx_file <path_to_smplx_data> --robot <path_to_robot_data> --save_path <path_to_save_robot_data.pkl> --rate_limit
# dataset
python scripts/smplx_to_robot_dataset.py --src_folder <path_to_dir_of_smplx_data> --tgt_folder <path_to_dir_to_save_robot_data> --robot <robot_name>
```

2. GMR redirected the actions to the joints of the target robot and organized the action data as follows:

```python
motion_data = {
    "fps": aligned_fps,
    "root_pos": root_pos,
    "root_rot": root_rot,
    "dof_pos": dof_pos,
    "local_body_pos": local_body_pos,
    "link_body_list": body_names,
}
```

If you are using a single-action redirection, `local_body_pos` will be `None`, but that's okay.

3. Run `motions/remakeGMRdata.py`, making sure to modify the original motion path and the desired storage path in the script. This will reorganize the motions as follows:

```python
data_dict = {
    "fps": fps,                                   # int64 scalar, sampling rate
    "dof_names": dof_names,                       # unicode array (D,)
    "body_names": body_names,                     # unicode array (B,)
    "dof_positions": dof_positions,               # float32 (N, D)
    "dof_velocities": dof_velocities_smoothed,    # float32 (N, D)
    "body_positions": body_positions,             # float32 (N, B, 3)
    "body_rotations": body_rotations,             # float32 (N, B, 4) (w,x,y,z)
    "body_linear_velocities": body_linear_velocities,     # float32 (N, B, 3)
    "body_angular_velocities": body_angular_velocities    # float32 (N, B, 3)
}
```

The position and rotation of the body are obtained by `pinocchio` through FK solving. If you are not using the UnitreeG1 robot, you need to provide the `urdf` asset file for the corresponding robot.

> This script is edited from [Humanoid-AMP](https://github.com/linden713/humanoid_amp)

4. If you are using a different robot, please modify the robot's assets in the environment and adjust the `num_amp_observation_space` in `AMPCfg`. You can run the program once, and the command line output of `IsaacLab` will display the dimension of `amp_obs`, which you can then fill in here.

## 💰 Adjust the ratio of the Reward

If you want the robot to achieve human-like movements while completing specific tasks, it is necessary to appropriately set the ratio between `style_reward` and `task_reward`. The algorithm parameters in this repository are located in `scripts/models`.