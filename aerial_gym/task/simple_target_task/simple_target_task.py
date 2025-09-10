from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np

from aerial_gym.utils.math import *
from aerial_gym.utils.dynamic_obs_controller import DynamicObsController

from aerial_gym.utils.logging import CustomLogger

import gymnasium as gym
from gym.spaces import Dict, Box

logger = CustomLogger("simple_target_task")


def dict_to_class(dict):
    return type("ClassFromDict", (object,), dict)


class SimpleTargetTask(BaseTask):
    def __init__(self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None):

        # overwrite the params if user has provided them
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp
        super().__init__(task_config)
        self.device = self.task_config.device
        
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )

        logger.info("Building environment for simple target task.")
        logger.info(
            "\nSim Name: {},\nEnv Name: {},\nRobot Name: {}, \nController Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )
        logger.info(
            "\nNum Envs: {},\nUse Warp: {},\nHeadless: {}".format(
                self.task_config.num_envs,
                self.task_config.use_warp,
                self.task_config.headless,
            )
        )

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            device=self.device,
            args=self.task_config.args,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        self.actions = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim),
            device=self.device,
            requires_grad=False,
        )
        self.lower_bound = None
        self.upper_bound = None
        self.dt = self.sim_env.sim_config.sim.dt

        self.update_bounds()

        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, dtype=torch.float32
        )
        self.target_velocity = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, dtype=torch.float32
        )
        self.obs_indice = self.find_asset_indices_by_type('dynamic_uav')
        self.update_obs_state()

        
        # 获取环境边界，留出安全边距
        safety_margin = 2.0  # 安全边距
        controller_min_pos = self.lower_bound + safety_margin
        controller_max_pos = self.upper_bound - safety_margin
        # 动态障碍物的最大速度
        max_obs_velocity = 2.0  # 可以根据需要调整
        
        self.dynamic_obs_controller = DynamicObsController(
            min_position=controller_min_pos,
            max_position=controller_max_pos,
            max_velocity=max_obs_velocity,
            num_envs=self.sim_env.num_envs,
            device=self.device,
            dt=self.dt,
            waypoint_update_freq=1000,  # 每200步更新一次路径点
            smoothing_factor=0.85,     # 速度平滑因子
            noise_scale=0.2            # 随机噪声强度
        )
            # self.dynamic_obs_controller = None


        self.obs_dict = self.sim_env.get_obs()
        self.obs_dict["num_obstacles_in_env"] = 1
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)
        
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.task_config.action_space_dim,),
            dtype=np.float32,
        )

        self.action_transformation_function = self.task_config.action_transformation_function

        self.prev_actions = torch.zeros_like(self.actions)
        self.counter = 0
        self.num_envs = self.sim_env.num_envs

        self.obs_twist = torch.zeros((self.sim_env.num_envs, self.sim_env.IGE_env.num_assets_per_env - 1, 6), device="cuda:0")
        self.infos = {}
        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (
                    self.sim_env.num_envs,
                    self.task_config.privileged_observation_space_dim,
                ),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
        }
    
    def update_bounds(self):
        self.lower_bound = self.sim_env.IGE_env.env_lower_bound
        self.upper_bound = self.sim_env.IGE_env.env_upper_bound

    def close(self):
        self.sim_env.delete_env()

    def update_obs_state(self):
        target_position_all = self.sim_env.get_obs_position()
        target_velocity_all = self.sim_env.get_obs_linvel()
        for i in range(self.sim_env.num_envs):
            self.target_position[i] = target_position_all[i, self.obs_indice[i][0]]
            self.target_velocity[i] = target_velocity_all[i, self.obs_indice[i][0]]

    def reset(self):
        self.sim_env.reset()
        self.update_obs_state()
        self.update_bounds()
        if self.dynamic_obs_controller is not None:
            # 获取新的边界，留出安全边距
            safety_margin = 2.0
            controller_min_pos = self.lower_bound + safety_margin
            controller_max_pos = self.upper_bound - safety_margin
            
            # 更新控制器边界
            self.dynamic_obs_controller.update_bounds(controller_min_pos, controller_max_pos)
            
            # 使用从环境获取的真实位置重置控制器
            self.dynamic_obs_controller.reset(initial_positions=self.target_position)

        self.infos = {}
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        self.sim_env.reset_idx(env_ids)
        self.update_obs_state()
        self.update_bounds()
        if self.dynamic_obs_controller is not None:
            # 获取新的边界，留出安全边距
            safety_margin = 2.0
            controller_min_pos = self.lower_bound + safety_margin
            controller_max_pos = self.upper_bound - safety_margin
            
            # 更新控制器边界
            self.dynamic_obs_controller.update_bounds(controller_min_pos, controller_max_pos)
            
            # 使用从环境获取的真实位置重置指定环境的控制器
            self.dynamic_obs_controller.reset(env_ids, initial_positions=self.target_position[env_ids])
        
        self.infos = {}
        return 

    def render(self):
        return self.sim_env.render()

    def step(self, actions):
        self.counter += 1
        self.prev_actions[:] = self.actions
        transformed_action = self.action_transformation_function(actions)
        self.actions = transformed_action
        self.compute_obs_next_action()

        self.sim_env.step(actions=self.actions, env_actions=self.obs_twist)
        self.update_obs_state()

        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)
        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()
        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )

        self.sim_env.post_reward_calculation_step()

        self.infos = {}  # self.obs_dict["infos"]

        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()

        return return_tuple
    
    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )
    
    def process_obs_for_task(self):
        self.task_obs["observations"][:, 0:3] = (
            self.target_position - self.obs_dict["robot_position"]
        )
        self.task_obs["observations"][:, 3:7] = self.obs_dict["robot_orientation"]
        # self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        # self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_angvel"]
        self.task_obs["observations"][:, 13:17] = self.target_velocity
        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def compute_obs_next_action(self):
        """计算动态障碍物的下一步动作"""
        self.obs_twist = torch.zeros((self.sim_env.num_envs, self.sim_env.IGE_env.num_assets_per_env - 1, 6), device="cuda:0")
        
        if self.dynamic_obs_controller is not None:
            # 获取当前动态障碍物的位置
            current_obs_positions = self.target_position.clone()  # 使用已更新的目标位置
            
            # 获取控制器输出的twist
            controller_twist = self.dynamic_obs_controller.get_twist(current_obs_positions)
            for env_idx in range(self.sim_env.num_envs):
                obs_asset_idx = self.obs_indice[env_idx][0]

                self.obs_twist[env_idx, obs_asset_idx] = controller_twist[env_idx]
                    

    def find_asset_indices_by_type(self, asset_type):
        """根据asset类型查找所有环境中的索引，返回二维列表[env_idx][asset_idx] = asset_id"""

        num_envs = len(self.sim_env.global_asset_dicts)
        all_indices = [[] for _ in range(num_envs)]
        
        for env_idx, asset_dicts in enumerate(self.sim_env.global_asset_dicts):
            for asset_idx, asset_dict in enumerate(asset_dicts):
                if asset_dict['asset_type'] == asset_type:
                    all_indices[env_idx].append(asset_idx)  
        return all_indices
    
    def compute_rewards_and_crashes(self, obs_dict):
        robot_position = obs_dict["robot_position"].to(dtype=torch.float32)
        target_position = self.target_position  # 使用已正确索引的目标位置
        robot_linvel = obs_dict["robot_linvel"]
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        robot_orientation = obs_dict["robot_orientation"]
        angular_velocity = obs_dict["robot_angvel"]

        pos_error_vehicle_frame = quat_apply_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )

        return compute_reward(
            pos_error_vehicle_frame,
            robot_linvel,
            robot_orientation,
            angular_velocity,
            obs_dict["crashes"],
            1.0, 
            self.actions,
            self.prev_actions,
            self.task_config.reward_parameters,
        )

@torch.jit.script
def exp_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * torch.exp(-exp * x * x)


@torch.jit.script
def exp_penalty_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * (torch.exp(-exp * x * x) - 1)


@torch.jit.script
def compute_reward(
    pos_error,
    lin_vels,
    robot_quats,
    robot_angvels,
    crashes,
    curriculum_level_multiplier,
    current_action,
    prev_actions,
    parameter_dict,
):  
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]

    dist = torch.norm(pos_error, dim=1)
    pos_reward = exp_func(dist, 3.0, 8.0) + exp_func(dist, 2.0, 4.0)
    dist_reward = (20 - dist) / 40.0

    ups = quat_axis(robot_quats, 2)
    
    up_reward = 2.0 * torch.sigmoid(5 * ups[..., 2]) - 1.0 
    up_reward = up_reward * 0.4  
    
    severe_flip_penalty = torch.where(
        ups[..., 2] < -0.4,
        -15.0 * torch.abs(ups[..., 2] + 0.4),  
        torch.zeros_like(ups[..., 2])
    )
    up_reward += severe_flip_penalty

    spinnage = torch.norm(robot_angvels, dim=1)
    ang_vel_reward = (1.0 / (1.0 + spinnage * spinnage)) * 3
    
    total_reward = (
        pos_reward + dist_reward + pos_reward * (up_reward + ang_vel_reward)
    )
    total_reward[:] = curriculum_level_multiplier * total_reward
    
    crashes[:] = torch.where(dist > 6.0, torch.ones_like(crashes), crashes)
    total_reward[:] = torch.where(crashes > 0.0, -20 * torch.ones_like(total_reward), total_reward)
    
    return total_reward, crashes