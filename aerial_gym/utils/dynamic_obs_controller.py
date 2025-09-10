import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class DynamicObsController:
    """
    动态障碍物控制器，使用样条曲线生成平滑且随机的飞行轨迹
    支持每个环境有不同的边界
    """
    
    def __init__(self, 
                 min_position: torch.Tensor,  # shape: (num_envs, 3) 或 (3,)
                 max_position: torch.Tensor,  # shape: (num_envs, 3) 或 (3,)
                 max_velocity: float,
                 num_envs: int,
                 device: str = "cuda:0",
                 dt: float = 0.01,
                 waypoint_update_freq: int = 200,  # 每200步更新一次waypoint
                 smoothing_factor: float = 0.85,   # 速度平滑因子
                 noise_scale: float = 0.2,         # 随机噪声强度
                 spline_points: int = 4):          # 样条曲线控制点数量
        
        self.num_envs = num_envs
        self.device = device

        min_position = min_position.to(device)
        max_position = max_position.to(device)
        
        if min_position.dim() == 1:  # shape: (3,)
            self.min_position = min_position.unsqueeze(0).expand(num_envs, 3)  # (num_envs, 3)
        elif min_position.dim() == 2:  # shape: (num_envs, 3)
            self.min_position = min_position
        else:
            raise ValueError(f"min_position should have shape (3,) or (num_envs, 3), got {min_position.shape}")
        
        if max_position.dim() == 1:  # shape: (3,)
            self.max_position = max_position.unsqueeze(0).expand(num_envs, 3)  # (num_envs, 3)
        elif max_position.dim() == 2:  # shape: (num_envs, 3)
            self.max_position = max_position
        else:
            raise ValueError(f"max_position should have shape (3,) or (num_envs, 3), got {max_position.shape}")
        
        
        self.max_velocity = max_velocity
        self.dt = dt
        self.waypoint_update_freq = waypoint_update_freq
        self.smoothing_factor = smoothing_factor
        self.noise_scale = noise_scale
        self.spline_points = spline_points
        
        self.current_positions = torch.zeros((num_envs, 3), device=device)
        self.current_velocities = torch.zeros((num_envs, 3), device=device)

        self.spline_control_points = torch.zeros((num_envs, spline_points, 3), device=device)

        self.trajectory_time = torch.zeros(num_envs, device=device)
        self.time_increment = 1.0 / waypoint_update_freq  

        self.trajectory_type = torch.randint(0, 3, (num_envs,), device=device)  # 0: 贝塞尔, 1: 圆形样条, 2: 螺旋样条
        self.trajectory_phase = torch.rand(num_envs, device=device) * 2 * np.pi
        self.trajectory_frequency = torch.rand(num_envs, device=device) * 0.02 + 0.01

        self.curve_amplitude = torch.rand(num_envs, device=device) * 2.0 + 1.0 
        self.curve_frequency = torch.rand(num_envs, device=device) * 0.5 + 0.3  

        self.step_counter = 0
        self.env_step_counters = torch.zeros(num_envs, device=device, dtype=torch.int32)

        self._initialize_positions()
        self._generate_spline_trajectory()
    
    def update_bounds(self, min_position: torch.Tensor, max_position: torch.Tensor):
        """更新边界并重新生成轨迹"""
        min_position = min_position.to(self.device)
        max_position = max_position.to(self.device)
        
        if min_position.dim() == 1:  # shape: (3,)
            self.min_position = min_position.unsqueeze(0).expand(self.num_envs, 3)
        else:  # shape: (num_envs, 3)
            self.min_position = min_position
            
        if max_position.dim() == 1:  # shape: (3,)
            self.max_position = max_position.unsqueeze(0).expand(self.num_envs, 3)
        else:  # shape: (num_envs, 3)
            self.max_position = max_position
        
        # 确保当前位置在新边界内
        self.current_positions = self._ensure_bounds(self.current_positions)
        
        # 重新生成轨迹
        self._generate_spline_trajectory()
    
    def _initialize_positions(self):
        """初始化随机位置 - 每个环境使用不同的边界"""
        for i in range(3):
            range_sizes = self.max_position[:, i] - self.min_position[:, i]
            min_positions = self.min_position[:, i]
            random_factors = torch.rand(self.num_envs, device=self.device)
            self.current_positions[:, i] = random_factors * range_sizes + min_positions
    
    def _generate_random_point_in_bounds(self, env_idx: int) -> torch.Tensor:
        """为指定环境在边界内生成随机点"""
        point = torch.zeros(3, device=self.device)
        for i in range(3):
            range_size = (self.max_position[env_idx, i] - self.min_position[env_idx, i]).item()
            min_pos = self.min_position[env_idx, i].item()
            
            point[i] = torch.rand(1, device=self.device).item() * range_size + min_pos
        return point
    
    def _generate_spline_trajectory(self):
        """生成样条曲线轨迹并预计算路径点"""
        # 随机生成新的轨迹类型和参数
        self.trajectory_type = torch.randint(0, 3, (self.num_envs,), device=self.device)
        self.curve_amplitude = torch.rand(self.num_envs, device=self.device) * 2.0 + 1.0
        self.curve_frequency = torch.rand(self.num_envs, device=self.device) * 0.5 + 0.3
        
        # 为每个环境生成样条控制点
        for env_idx in range(self.num_envs):
            traj_type = self.trajectory_type[env_idx].item()
            
            if traj_type == 0:  # 贝塞尔曲线
                self._generate_bezier_points(env_idx)
            elif traj_type == 1:  # 圆形样条
                self._generate_circular_spline(env_idx)
            else:  # 螺旋样条
                self._generate_spiral_spline(env_idx)
        
        # 重置参数
        self.trajectory_phase = torch.rand(self.num_envs, device=self.device) * 2 * np.pi
    
    def _generate_bezier_points(self, env_idx: int):
        """生成贝塞尔曲线控制点"""
        # 起点是当前位置
        self.spline_control_points[env_idx, 0] = self.current_positions[env_idx]
        
        # 随机生成中间控制点
        for i in range(1, self.spline_points - 1):
            for dim in range(3):
                # 在当前位置附近生成控制点，增加变化
                center = self.current_positions[env_idx, dim].item()
                range_size = ((self.max_position[env_idx, dim] - self.min_position[env_idx, dim]) * 0.6).item()
                offset = (torch.rand(1, device=self.device).item() - 0.5) * range_size
                
                point = center + offset
                min_bound = self.min_position[env_idx, dim].item()
                max_bound = self.max_position[env_idx, dim].item()
                point = max(min_bound, min(max_bound, point))
                
                self.spline_control_points[env_idx, i, dim] = point
        
        # 终点随机生成
        end_point = self._generate_random_point_in_bounds(env_idx)
        self.spline_control_points[env_idx, -1] = end_point
    
    def _generate_circular_spline(self, env_idx: int):
        """生成圆形样条轨迹"""
        # 圆心 - 使用当前环境的边界
        center = torch.zeros(3, device=self.device)
        for dim in range(3):
            center[dim] = (self.max_position[env_idx, dim] + self.min_position[env_idx, dim]) / 2
        
        # 随机半径（在安全范围内）
        env_ranges = self.max_position[env_idx] - self.min_position[env_idx]
        max_radius = torch.min(env_ranges / 3).item()
        radius = torch.rand(1, device=self.device).item() * max_radius * 0.8
        
        # 随机倾斜角度
        tilt_x = (torch.rand(1, device=self.device).item() - 0.5) * 0.4
        tilt_y = (torch.rand(1, device=self.device).item() - 0.5) * 0.4
        
        # 生成圆形轨迹点
        for i in range(self.spline_points):
            angle = 2 * np.pi * i / self.spline_points + self.trajectory_phase[env_idx].item()
            
            # 基础圆形
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = radius * np.sin(angle * 2) * 0.3  # Z轴上的波动
            
            # 应用倾斜
            self.spline_control_points[env_idx, i, 0] = center[0] + x + y * tilt_x
            self.spline_control_points[env_idx, i, 1] = center[1] + y + x * tilt_y
            self.spline_control_points[env_idx, i, 2] = center[2] + z
        
        # 确保在边界内
        self.spline_control_points[env_idx] = self._ensure_bounds_single_env(self.spline_control_points[env_idx], env_idx)
    
    def _generate_spiral_spline(self, env_idx: int):
        """生成螺旋样条轨迹"""
        # 螺旋中心
        center = torch.zeros(3, device=self.device)
        for dim in range(3):
            range_size = (self.max_position[env_idx, dim] - self.min_position[env_idx, dim]).item()
            min_pos = self.min_position[env_idx, dim].item()
            
            center[dim] = torch.rand(1, device=self.device).item() * range_size * 0.6 + \
                         min_pos + range_size * 0.2
        
        # 螺旋参数
        initial_radius = torch.rand(1, device=self.device).item() * 2.0 + 0.5
        radius_decay = 0.8
        height_step = ((self.max_position[env_idx, 2] - self.min_position[env_idx, 2]) / self.spline_points * 0.3).item()
        
        for i in range(self.spline_points):
            t = i / (self.spline_points - 1)
            angle = 4 * np.pi * t + self.trajectory_phase[env_idx].item()
            radius = initial_radius * (radius_decay ** t)
            
            self.spline_control_points[env_idx, i, 0] = center[0] + radius * np.cos(angle)
            self.spline_control_points[env_idx, i, 1] = center[1] + radius * np.sin(angle)
            self.spline_control_points[env_idx, i, 2] = center[2] + height_step * (i - self.spline_points/2)
        
        # 确保在边界内
        self.spline_control_points[env_idx] = self._ensure_bounds_single_env(self.spline_control_points[env_idx], env_idx)
    
    def _ensure_bounds_single_env(self, positions: torch.Tensor, env_idx: int) -> torch.Tensor:
        """确保单个环境的位置在其边界内"""
        return torch.clamp(positions, self.min_position[env_idx], self.max_position[env_idx])
    
    def _get_current_spline_position(self, env_idx: int, t: float) -> torch.Tensor:
        """根据样条参数获取当前位置"""
        control_points = self.spline_control_points[env_idx]
        traj_type = self.trajectory_type[env_idx].item()
        
        if traj_type == 0 and control_points.shape[0] == 4:  # 贝塞尔曲线
            return self._cubic_bezier_interpolation(control_points, t)
        else:  # 其他类型使用Catmull-Rom样条
            return self._catmull_rom_interpolation(control_points, t)
    
    def _cubic_bezier_interpolation(self, control_points: torch.Tensor, t: float) -> torch.Tensor:
        """三次贝塞尔曲线插值"""
        if control_points.shape[0] != 4:
            # 如果不是4个点，使用线性插值作为fallback
            return self._linear_spline_interpolation(control_points, t)
        
        # 贝塞尔曲线公式
        t_tensor = torch.tensor(t, device=self.device)
        one_minus_t = 1.0 - t_tensor
        
        coeff0 = one_minus_t ** 3
        coeff1 = 3 * one_minus_t ** 2 * t_tensor
        coeff2 = 3 * one_minus_t * t_tensor ** 2
        coeff3 = t_tensor ** 3
        
        result = (coeff0 * control_points[0] + 
                 coeff1 * control_points[1] + 
                 coeff2 * control_points[2] + 
                 coeff3 * control_points[3])
        
        return result
    
    def _linear_spline_interpolation(self, control_points: torch.Tensor, t: float) -> torch.Tensor:
        """线性样条插值"""
        n_segments = control_points.shape[0] - 1
        segment_t = t * n_segments
        segment_idx = int(torch.clamp(torch.floor(torch.tensor(segment_t)), 0, n_segments - 1))
        local_t = segment_t - segment_idx
        
        if segment_idx >= n_segments:
            return control_points[-1]
        
        return control_points[segment_idx] * (1 - local_t) + control_points[segment_idx + 1] * local_t
    
    def _catmull_rom_interpolation(self, control_points: torch.Tensor, t: float) -> torch.Tensor:
        """Catmull-Rom样条插值（更平滑）"""
        n_points = control_points.shape[0]
        if n_points < 4:
            return self._linear_spline_interpolation(control_points, t)
        
        # 扩展控制点以支持Catmull-Rom
        extended_points = torch.zeros((n_points + 2, 3), device=self.device)
        extended_points[0] = 2 * control_points[0] - control_points[1]  # 外推第一个点
        extended_points[1:-1] = control_points
        extended_points[-1] = 2 * control_points[-1] - control_points[-2]  # 外推最后一个点
        
        # 计算segment
        segment_t = t * (n_points - 1)
        segment_idx = int(torch.clamp(torch.floor(torch.tensor(segment_t)), 0, n_points - 2))
        local_t = segment_t - segment_idx
        
        # Catmull-Rom公式
        p0 = extended_points[segment_idx]
        p1 = extended_points[segment_idx + 1]
        p2 = extended_points[segment_idx + 2]
        p3 = extended_points[segment_idx + 3]
        
        t_tensor = torch.tensor(local_t, device=self.device)
        t2 = t_tensor * t_tensor
        t3 = t2 * t_tensor
        
        result = 0.5 * ((2 * p1) +
                       (-p0 + p2) * t_tensor +
                       (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                       (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
        
        return result
    
    def _ensure_bounds(self, positions: torch.Tensor) -> torch.Tensor:
        """确保位置在边界内 - 支持每个环境不同的边界"""
        return torch.clamp(positions, self.min_position, self.max_position)
    
    def _limit_velocity(self, velocities: torch.Tensor) -> torch.Tensor:
        """限制速度大小"""
        vel_norm = torch.norm(velocities, dim=1, keepdim=True)
        scale = torch.where(vel_norm > self.max_velocity, 
                          self.max_velocity / torch.clamp(vel_norm, min=1e-6), 
                          torch.ones_like(vel_norm))
        return velocities * scale
    
    def get_twist(self, current_obs_positions: torch.Tensor) -> torch.Tensor:
        """
        获取twist控制命令 - 使用样条参数计算
        
        Args:
            current_obs_positions: 当前障碍物位置 shape: (num_envs, 3)
        
        Returns:
            twist: shape: (num_envs, 6) [vx, vy, vz, wx, wy, wz]
        """
        self.step_counter += 1
        self.env_step_counters += 1
        
        # 更新当前位置
        self.current_positions = current_obs_positions.clone()
        
        # 检查哪些环境需要重新生成轨迹
        trajectory_finished_mask = self.trajectory_time >= 1.0
        force_update_mask = (self.env_step_counters % self.waypoint_update_freq == 0)
        update_mask = trajectory_finished_mask | force_update_mask
        
        if update_mask.any():
            update_indices = torch.where(update_mask)[0]
            
            # 为需要更新的环境重新生成轨迹
            for env_idx in update_indices:
                env_idx = env_idx.item()
                traj_type = torch.randint(0, 3, (1,), device=self.device).item()
                self.trajectory_type[env_idx] = traj_type
                
                if traj_type == 0:  # 贝塞尔曲线
                    self._generate_bezier_points(env_idx)
                elif traj_type == 1:  # 圆形样条
                    self._generate_circular_spline(env_idx)
                else:  # 螺旋样条
                    self._generate_spiral_spline(env_idx)
            
            # 重置相关参数
            self.trajectory_time[update_indices] = 0.0
            self.env_step_counters[update_indices] = 0
        
        # 使用样条参数直接计算当前和前瞻位置
        current_targets = torch.zeros_like(self.current_positions)
        lookahead_targets = torch.zeros_like(self.current_positions)
        
        for env_idx in range(self.num_envs):
            current_t = self.trajectory_time[env_idx].item()
            current_targets[env_idx] = self._get_current_spline_position(env_idx, current_t)
            
            # 前向查看计算切线方向
            lookahead_t = min(current_t + 0.05, 1.0)  # 前瞻5%的轨迹长度
            lookahead_targets[env_idx] = self._get_current_spline_position(env_idx, lookahead_t)
        
        # 确保位置在边界内
        current_targets = self._ensure_bounds(current_targets)
        lookahead_targets = self._ensure_bounds(lookahead_targets)
        
        # 更新轨迹时间参数
        self.trajectory_time += self.time_increment
        self.trajectory_time = torch.clamp(self.trajectory_time, 0.0, 1.0)
        
        # 计算期望速度（沿轨迹切线方向）
        direction_to_target = lookahead_targets - current_targets
        direction_norm = torch.norm(direction_to_target, dim=1, keepdim=True)
        direction_to_target = torch.where(direction_norm > 1e-6, 
                                        direction_to_target / direction_norm, 
                                        torch.zeros_like(direction_to_target))
        
        # 基础速度
        base_speed = self.max_velocity * 0.6  # 使用最大速度的60%
        desired_velocity = direction_to_target * base_speed
        
        # 添加随机噪声
        noise = torch.randn_like(desired_velocity) * self.noise_scale * self.max_velocity
        desired_velocity += noise
        
        # 平滑速度变化
        self.current_velocities = (self.smoothing_factor * self.current_velocities + 
                                 (1 - self.smoothing_factor) * desired_velocity)
        
        # 限制速度
        self.current_velocities = self._limit_velocity(self.current_velocities)
        
        # 边界碰撞检测和处理 - 每个环境使用不同的边界
        next_positions = self.current_positions + self.current_velocities * self.dt
        
        for i in range(3):
            # 检查边界碰撞 - 使用每个环境的边界
            will_exceed_min = next_positions[:, i] < self.min_position[:, i]
            will_exceed_max = next_positions[:, i] > self.max_position[:, i]
            
            # 边界反弹
            self.current_velocities[will_exceed_min, i] = torch.abs(self.current_velocities[will_exceed_min, i]) * 0.8
            self.current_velocities[will_exceed_max, i] = -torch.abs(self.current_velocities[will_exceed_max, i]) * 0.8
            
            # 如果碰撞，强制重新生成轨迹
            collision_mask = will_exceed_min | will_exceed_max
            if collision_mask.any():
                collision_indices = torch.where(collision_mask)[0]
                for env_idx in collision_indices:
                    env_idx = env_idx.item()
                    # 重新生成单个环境的轨迹
                    traj_type = torch.randint(0, 3, (1,), device=self.device).item()
                    self.trajectory_type[env_idx] = traj_type
                    
                    if traj_type == 0:  # 贝塞尔曲线
                        self._generate_bezier_points(env_idx)
                    elif traj_type == 1:  # 圆形样条
                        self._generate_circular_spline(env_idx)
                    else:  # 螺旋样条
                        self._generate_spiral_spline(env_idx)
                
                # 重置碰撞环境的时间参数
                self.trajectory_time[collision_indices] = 0.0
        
        # 生成角速度（根据运动方向调整偏航）
        angular_velocity = torch.randn((self.num_envs, 3), device=self.device) * 0.1
        
        # 根据速度方向计算偏航角速度
        vel_norm = torch.norm(self.current_velocities[:, :2], dim=1, keepdim=True)
        yaw_rate = torch.where(vel_norm > 0.1, 
                              torch.atan2(self.current_velocities[:, 1:2], self.current_velocities[:, 0:1]) * 0.1,
                              torch.zeros_like(vel_norm))
        angular_velocity[:, 2:3] += yaw_rate
        
        # 构造twist（线速度 + 角速度）
        twist = torch.cat([self.current_velocities, angular_velocity], dim=1)
        
        return twist
    
    def reset(self, env_ids: Optional[torch.Tensor] = None, initial_positions: Optional[torch.Tensor] = None):
        """重置指定环境的状态"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # 如果提供了初始位置，使用它们；否则随机生成
        if initial_positions is not None:
            self.current_positions[env_ids] = initial_positions
        else:
            # 重新初始化指定环境的位置（随机化）- 使用每个环境的边界
            for i in range(3):
                range_sizes = self.max_position[env_ids, i] - self.min_position[env_ids, i]
                min_positions = self.min_position[env_ids, i]
                
                random_factors = torch.rand(len(env_ids), device=self.device)
                self.current_positions[env_ids, i] = random_factors * range_sizes + min_positions
        
        # 重置状态
        self.current_velocities[env_ids] = 0.0
        self.trajectory_time[env_ids] = 0.0  # 重置轨迹时间参数
        self.env_step_counters[env_ids] = 0
        
        # 重新生成指定环境的轨迹控制点
        for env_idx in env_ids:
            env_idx = env_idx.item()
            traj_type = torch.randint(0, 3, (1,), device=self.device).item()
            self.trajectory_type[env_idx] = traj_type
            
            if traj_type == 0:  # 贝塞尔曲线
                self._generate_bezier_points(env_idx)
            elif traj_type == 1:  # 圆形样条
                self._generate_circular_spline(env_idx)
            else:  # 螺旋样条
                self._generate_spiral_spline(env_idx)
        
        # 重置轨迹参数
        self.trajectory_phase[env_ids] = torch.rand(len(env_ids), device=self.device) * 2 * np.pi
        self.curve_amplitude[env_ids] = torch.rand(len(env_ids), device=self.device) * 2.0 + 1.0
        self.curve_frequency[env_ids] = torch.rand(len(env_ids), device=self.device) * 0.5 + 0.3