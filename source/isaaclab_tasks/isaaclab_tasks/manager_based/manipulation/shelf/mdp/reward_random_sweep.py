from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from dataclasses import MISSING


from isaaclab.assets import RigidObject, Articulation, RigidObjectCollection, AssetBase
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat, euler_xyz_from_quat, subtract_frame_transforms
from isaaclab_tasks.manager_based.manipulation.shelf.src.shelf_utils import normalize_angle

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    

def reward_for_hand_reaching(env: ManagerBasedRLEnv,
                            object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
                            ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
                            wrist_frame_cfg: SceneEntityCfg=SceneEntityCfg("wrist_frame")):
    
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]
    wrist: FrameTransformer = env.scene[wrist_frame_cfg.name]
    

    # Get the target IDs directly from the environment tensor
    target_ids = env.target_id.squeeze(-1).long()  # Shape: (num_envs,)
    target_width = env.target_width
    sweep_dir = torch.sign(env.sweep_dir[:, 1])

    # Get the world state(position, orientation, linear velocity, angular velocity); R^13
    target_pos_w = object_collection.data.object_pos_w[torch.arange(env.scene.num_envs), target_ids].clone()
    ee_pos_w = ee.data.target_pos_w.clone()
    wrist_pos_w = wrist.data.target_pos_w.clone()


    offset_pos = target_pos_w.clone()
    offset_pos[:, 0] = offset_pos[:, 0] - 0.02
    offset_pos[:, 1] = offset_pos[:, 1] - target_width[:, 0] * sweep_dir
    offset_pos[:, 2] = offset_pos[:, 2] + 0.09
      

    distance_ee = torch.norm((offset_pos[:, :3] - ee_pos_w[..., 0, :3]), dim=-1, p=2)
    distance_wrist = torch.norm((offset_pos[:, 1:3] - wrist_pos_w[..., 0, 1:3]), dim=-1, p=2)
    distance = distance_ee + distance_wrist

    alpha = -10.0 
    reward = torch.exp(alpha * distance_ee)

    return reward

def align_ee_target(env: ManagerBasedRLEnv,
                     shelf_cfg: SceneEntityCfg = SceneEntityCfg("shelf"),
                     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    shelf: RigidObject = env.scene[shelf_cfg.name]
    
    shelf_defualt_quat_w = shelf.data.default_root_state[:, 3:7]
    ee_tcp_quat = ee_frame.data.target_quat_w.clone()
    
    shelf_rot_mat = matrix_from_quat(shelf_defualt_quat_w)
    ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat[..., 0 , :])
    
    ee_tcp_y = ee_tcp_rot_mat[..., 1]
    shelf_z = shelf_rot_mat[..., 2] 
    
    align_y_z = torch.bmm(ee_tcp_y.unsqueeze(1), shelf_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    return torch.sign(align_y_z) * align_y_z**2 


def pushing_target(env: ManagerBasedRLEnv, 
                   command_name: str,
                   object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),):
    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]
    ee_frame_cfg = SceneEntityCfg("ee_frame")
    wrist_frame_cfg = SceneEntityCfg("wrist_frame")

    robot: Articulation = env.scene[asset_cfg.name]

    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    wrist: FrameTransformer = env.scene[wrist_frame_cfg.name]

    command = env.command_manager.get_command(command_name)

    # obtain the desired and current positions
    des_pos_w = command[:, :3]
    # Get the target IDs directly from the environment tensor
    target_ids = env.target_id.squeeze(-1).long()  # Shape: (num_envs,)
    target_width = env.target_width
    sweep_dir = torch.sign(env.sweep_dir[:, 1])

    # Get the world state(position, orientation, linear velocity, angular velocity); R^13
    target_pos_w = object_collection.data.object_pos_w[torch.arange(env.scene.num_envs), target_ids]
    target_lin_vel_w = object_collection.data.object_lin_vel_w[torch.arange(env.scene.num_envs), target_ids]

    ee_pos_w = ee.data.target_pos_w[..., 0, :]
    wrist_pos_w = wrist.data.target_pos_w[..., 0, :].clone()

    offset_pos = target_pos_w.clone()
    offset_pos[:, 0] = offset_pos[:, 0] - 0.02
    offset_pos[:, 1] = offset_pos[:, 1] - target_width[:, 0] * sweep_dir
    offset_pos[:, 2] = offset_pos[:, 2] + 0.09

    distance = torch.norm((des_pos_w - target_pos_w), dim=-1, p=2)
    zeta_m = torch.where((torch.norm(offset_pos - ee_pos_w, dim=-1, p=2)) < 0.04 , torch.where(torch.abs(offset_pos[:, 1] - wrist_pos_w[:, 1])<0.04, 1, 0), 0)
    obj_vel_rew = torch.where(torch.abs(target_lin_vel_w[:, 1]) > 0.05, torch.where(torch.abs(target_lin_vel_w[:, 1]) < 0.1, 0.5, -0.5), 0)
    reward = torch.where(distance < 0.03, 2.0 * (1 - distance/0.18), zeta_m *((1 - distance/0.18) + obj_vel_rew))
    return reward

def homing_reward(env: ManagerBasedRLEnv,
                  command_name: str,
                  object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                  shelf_cfg: SceneEntityCfg = SceneEntityCfg("shelf"),
                  ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")):
    robot: Articulation = env.scene[asset_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]


    command = env.command_manager.get_command(command_name)

    target_ids = env.target_id.squeeze(-1).long()  # Shape: (num_envs,)

    # Get the world state(position, orientation, linear velocity, angular velocity); R^13
    target_pos_w = object_collection.data.object_pos_w[torch.arange(env.scene.num_envs), target_ids]


    # obtain the desired and current positions
    des_pos_w = command[:, :3]
    distance = torch.norm((des_pos_w[:, 1:] - target_pos_w[:, 1:]), dim=-1, p=2)
    joint_pos_error = torch.sum(torch.abs(robot.data.joint_pos[:, : 5] - robot.data.default_joint_pos[:, :5]), dim=1)
    reward_for_home_pose = torch.exp(-0.5 * joint_pos_error)
    
    return torch.where(distance < 0.03, reward_for_home_pose, 0)

    
def object_collision(env: ManagerBasedRLEnv,
                object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),)-> torch.Tensor:
    
    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]

    # Get the target IDs directly from the environment tensor
    target_ids = env.target_id.squeeze(-1).long()  # Shape: (num_envs,)

    objects_lin_vel_w = object_collection.data.object_lin_vel_w.clone()
    objects_lin_vel_w = torch.round(objects_lin_vel_w, decimals=2)

    objects_lin_vel_w[torch.arange(env.scene.num_envs), target_ids,:] = 0.0
    
    
    reward = torch.tanh(torch.sum(torch.abs(objects_lin_vel_w)))
    return reward

     
class shelf_Collision(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        ee_frame_cfg = SceneEntityCfg("ee_frame")
        shelf_cfg = SceneEntityCfg("shelf")
        wrist_frame_cfg= SceneEntityCfg("wrist_frame")
        finger_frame_cfg = SceneEntityCfg("finger_frame")

        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]
        self._finger: FrameTransformer = env.scene[finger_frame_cfg.name]
        self._shelf: RigidObject = env.scene[shelf_cfg.name]
        self._wrist: FrameTransformer = env.scene[wrist_frame_cfg.name]
        self._initial_shelf_pos = self._shelf.data.default_root_state[:, :3] + env.scene.env_origins


    
    def __call__(self, env: ManagerBasedRLEnv,):

        collision = self.shelf_collision_penalty(env)
        collision_dynamic = self.shelf_dynamic_penalty(env)
        return collision + collision_dynamic

    def shelf_collision_penalty(self,env: ManagerBasedRLEnv,) -> torch.Tensor:

        shelf_vel = self._shelf.data.root_vel_w
        shelf_delta = self._shelf.data.root_pos_w - self._initial_shelf_pos

        moved = torch.where((torch.norm(shelf_delta , dim=-1, p=2) + torch.norm(shelf_vel , dim=-1, p=2))> 0.005, 1.0, 0.0)
        return moved

    def shelf_dynamic_penalty(self, env: ManagerBasedRLEnv,) -> torch.Tensor:
        shelf_pos_w = self._shelf.data.root_pos_w .clone()
        shelf_pos_w[:,2] = shelf_pos_w[:, 2] + 1.06

        distance = torch.norm(shelf_pos_w - self._ee.data.target_pos_w[..., 0, :], dim=-1, p=2)
        zeta = torch.where(distance < 0.2, 1, 0)
        dst_l_shelf = self._finger.data.target_pos_w[..., 0, 2] - (shelf_pos_w[:,2])
        dst_r_shelf = self._finger.data.target_pos_w[..., 1, 2] - (shelf_pos_w[:,2])
        dst_wrist_shelf = self._wrist.data.target_pos_w[..., 0, 2] - (shelf_pos_w[:,2])


        reward_l = 1 - dst_l_shelf / 0.02
        reward_r = 1 - dst_r_shelf / 0.02
        reward_wrist = 1 - dst_wrist_shelf / 0.08


        reward_l = torch.clamp(reward_l, 0, 1)
        reward_r = torch.clamp(reward_r, 0, 1)
        reward_wrist = torch.clamp(reward_wrist, 0, 1)

        R = zeta * (reward_l + reward_r + reward_wrist)

        return R
    
    

def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    return torch.sum(torch.square(asset.data.joint_vel[:, :6]), dim=1)