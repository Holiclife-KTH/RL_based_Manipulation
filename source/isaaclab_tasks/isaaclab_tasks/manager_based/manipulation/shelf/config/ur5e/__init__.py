import gymnasium as gym
import os

from . import agents

gym.register(
    id="Isaac-Sweep-Object-UR5e-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR5eShelfEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR5eSweepPPORunnerCfg",
    },
)