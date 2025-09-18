# minimal_g1_amp

这个仓库是一个在 [IsaacLab](https://github.com/isaac-sim/IsaacLab) 中通过ManagerBasedRL环境实现人形机器人类人步态行走的最小示例，仓库基于 `IsaacLab` 中的 `Humanoid-28-AMP` 任务修改。

> 部分代码来自 [Humanoid-AMP](https://github.com/linden713/humanoid_amp)

This repository is a minimal example of achieving humanoid walking in the [IsaacLab](https://github.com/isaac-sim/IsaacLab) environment through Manager-Based Reinforcement Learning. It is modified based on the `Humanoid-28-AMP` task in `IsaacLab`.

> Part of the code comes from [Humanoid-AMP](https://github.com/linden713/humanoid_amp)

[English Document](README_EN.md)

## 🚀 快速上手

1. 将任务链接到 `IsaacLab` 仓库下，这是为了能快速将环境注册到 `isaaclab.tasks` 中

```bash
ln -s minimal_g1_amp/amp_task YourPATHOFISAACLAB/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/g1_amp
```

2. 执行训练脚本

```bash
python scripts/train_amp.py
```

> 这个训练脚本是按照 `skrl` 教程中推荐的写法组织的，你也可以按照 `IsaacLab.scripts.reinforcement_learning` 中的习惯来组织代码

3. 训练完成后录制视频

```bash
python scripts/play_amp.py
```

之后可以在 `runs` 中找到刚刚的实验以及视频

4. 学习跳舞任务

在 `scripts/train_amp.py` 和 `scripts/play_amp.py` 中，将

```python
# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-G1-AMP-Walk-v0", cli_args=["--headless"],)
```

修改为

```python
# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-G1-AMP-Dance-v0", cli_args=["--headless"],)
```

这实际上是在 `amp_task/g1_amp_env_cfg.py` 中继承了 Walk 的任务，然后替换了amp的动作文件，并在 `amp_task/__init__.py` 中注册为了一个新的task。你也可以按照这种方式实现你自己的task。

这个动作文件来自 `LAFAN` 数据集的 `dance1_subject1`.


## 🤖 使用自己的AMP数据

高质量的模仿数据对AMP算法的学习过程非常重要，在本仓库中，你可以修改 `amp_task/g1_amp_env_cfg.py` 中 `AMPCfg` 中的 `motion_file` 来更改动作数据文件。

### Unitree机器人

可以直接从HuggingFace下载重定向后的数据集

- [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
- [ember-lab-berkeley/AMASS_Retargeted_for_G1](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1)

### GMR重定向

也可以使用 [GMR](https://github.com/YanjieZe/GMR) 将原始的AMASS或LAFAN数据集重定向并转换为 本仓库所需的`npz` 数据集

1. 首先按照 [GMR](https://github.com/YanjieZe/GMR) 的方式重定向动作文件到自己的机器人

- 下载LaFan数据集
- 安装 [GMR](https://github.com/YanjieZe/GMR)
- 运行 [GMR](https://github.com/YanjieZe/GMR) 的重定向流程

```bash
# single motion
python scripts/smplx_to_robot.py --smplx_file <path_to_smplx_data> --robot <path_to_robot_data> --save_path <path_to_save_robot_data.pkl> --rate_limit
# dataset
python scripts/smplx_to_robot_dataset.py --src_folder <path_to_dir_of_smplx_data> --tgt_folder <path_to_dir_to_save_robot_data> --robot <robot_name>
```

2. GMR将动作重定向到了所需机器人的关节上，并把动作数据组织为：
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

如果你使用的是单个动作的重定向，`local_body_pos` 将会是 `None`，但这没关系。

3. 运行 `motions/remakeGMRdata.py`，注意修改脚本中原始动作的路径和希望的存储路径。这会将动作重新组织为：
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

其中身体的位置、旋转由 `pinocchio` 通过FK求解得到。如果你使用的不是UnitreeG1机器人，需要提供相应机器人的 `urdf` 资产文件。

> 该脚本修改自[Humanoid-AMP](https://github.com/linden713/humanoid_amp)

4. 如果你使用了其他的机器人，请修改环境中机器人的资产，同时修改 `AMPCfg` 中的 `num_amp_observation_space`。可以先运行一遍，`IsaacLab`的命令行输出中会显示 `amp_obs` 的维度，填到这里即可。

## 💰 修改Reward的比例

如果希望机器人在实现类人动作的同时完成具体的task，需要适当设置 `style_reward` 与 `task_reward` 的比例。本仓库中的算法参数位于 `scripts/models`。