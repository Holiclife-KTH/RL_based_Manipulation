from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg
from isaaclab.devices import DevicesCfg
from isaaclab.devices.gamepad import Se3GamepadCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.shelf.mdp as mdp

##
# Scene definition
##


@configclass
class ShelfSweepRandomSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""
    
    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    # mount = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Mount",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"omniverse://localhost/Library/Shelf/Arena/thor_table.usd",
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.79505), rot=(1.0, 0.0, 0.0, 0.0),),
    # )
    
    shelf = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=sim_utils.UsdFileCfg(usd_path=f"omniverse://localhost/Library/Shelf/Arena/speedrack.usd", mass_props=MassPropertiesCfg(mass=100),),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.7, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        debug_vis=False,
    )


    # robots
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    finger_frame: FrameTransformerCfg = MISSING
    wrist_frame: FrameTransformerCfg = MISSING

    #Objects
    object_collection: RigidObjectCollectionCfg = MISSING


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    target_goal_pos = mdp.DynamicObjectGoalPosCommandCfg(
        asset_name=MISSING,
        asset_dict=MISSING,
        object_id_dict_rev=MISSING,
        init_pos_offset=(0.0, 0.18, 0.0),
        debug_vis=True,)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.MA_joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.MA_joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        target_obs_state = ObsTerm(func=mdp.MA_object_position_in_RRF, params={}, noise = Unoise(n_min=-0.01, n_max=0.01))
        target_obj_width = ObsTerm(func=mdp.MA_object_width, noise = Unoise(n_min=-0.01, n_max=0.01))
        ee_pose = ObsTerm(func=mdp.ee_pos_r)
        goal_pos = ObsTerm(func=mdp.MA_target_goal_command, params={"command_name": MISSING})


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    object_spawn = EventTerm(func=mdp.randomize_scene, 
                             params={"asset_dict": MISSING, "pose_array":MISSING, "object_width_dict": MISSING,"ceiling_height": MISSING, "task_mode": MISSING}, mode="reset")



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # action penalty
    # Tagucci: -0.005 / -0.01 / -0.03
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_vel = RewTerm(
        func=mdp.reward_random_sweep.joint_vel_l2,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Tagucci: -0.01 / -0.1 / -0.5
    shelf_collision = RewTerm(func=mdp.reward_random_sweep.shelf_Collision, params={}, weight=-0.01)

    object_collision = RewTerm(func=mdp.reward_random_sweep.object_collision, params={}, weight=-0.01)


    # Tagucci: 1.0 / 2.0 / 3.0
    reaching = RewTerm(
        func=mdp.reward_random_sweep.reward_for_hand_reaching,
        weight=1.0,
        params={}
    )

    # Tagucci: 1.0 / 2.0 / 3.0
    orientation = RewTerm(
        func=mdp.reward_random_sweep.align_ee_target,
        weight=1.0,
        params={},
    )
    # Tagucci: 3.0 / 6.0 / 9.0
    sweeping_object = RewTerm(func=mdp.reward_random_sweep.pushing_target, 
                              params={"command_name": "target_goal_pos"}, 
                              weight=3.0) # 14 

    # Tagucci: 9.0 / 12.0 / 15.0
    homing_after_sweep = RewTerm(func=mdp.reward_random_sweep.homing_reward, params={"command_name": "target_goal_pos"}, weight=9.0) #12 

    

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_drop = DoneTerm(func=mdp.drop_object_termination, time_out=False, params={"height_condition":MISSING})
    push_fast = DoneTerm(func=mdp.push_fast_termination, time_out=False, params={"speed_condition":MISSING})
    shelf_collision = DoneTerm(func=mdp.shelf_collision_termination,time_out=False, params={"threshold": 0.1})
    hand_velocity = DoneTerm(func=mdp.hand_velocity_termination, time_out=False, params={"threshold": 1.0})

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    obj_collision = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "object_collision", "weight": -0.5, "num_steps": 250000}
    )
    
    # homing = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "homing_after_sweep", "weight": 12.0, "num_steps": 300000}
    # )



##
# Environment configuration
##


@configclass
class ShelfSweepRandomEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ShelfSweepRandomSceneCfg = ShelfSweepRandomSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.render_interval = 2
        # simulation settings
        self.sim.dt = 0.01  # 100Hz

        self.sim.physx .bounce_threshold_velocity = 0.2
        # self.sim.physx.bounce_threshold_velocity = 0.01

        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 16 * 16
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 * 16
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2 ** 17

        
