import torch

pi = 3.1415926

class task_config:
    seed = -1
    sim_name = "base_sim"
    env_name = "simple_target_env"
    robot_name = "x500"
    controller_name = "lee_velocity_control"
    args = {}
    num_envs = 4096
    use_warp = False
    headless = False
    device = "cuda:0"
    # self.pos_error, self.q, self.vel, self.ang_vel, obs.vel
    observation_space_dim = 16
    privileged_observation_space_dim = 0
    action_space_dim = 4
    episode_len_steps = 1000  # real physics time for simulation is this value multiplied by sim.dt
    return_state_before_reset = False
    reward_parameters = {
        "pos_error_gain1": [2.0, 2.0, 2.0],
        "pos_error_exp1": [1 / 3.5, 1 / 3.5, 1 / 3.5],
        "pos_error_gain2": [2.0, 2.0, 2.0],
        "pos_error_exp2": [2.0, 2.0, 2.0],
        "dist_reward_coefficient": 7.5,
        "max_dist": 15.0,
        "action_diff_penalty_gain": [1.0, 1.0, 1.0],
        "absolute_action_reward_gain": [2.0, 2.0, 2.0],
        "crash_penalty": -100,
    }

    def action_transformation_function(action):
        # wait to add action transformation function
        action[:, 0] = torch.clamp(action[:, 0], -1.0, 1.0)
        action[:, 1] = torch.clamp(action[:, 1], -1.0, 1.0)
        action[:, 2] = torch.clamp(action[:, 2], -1.0, 1.0)
        action[:, 3] = torch.clamp(action[:, 3], -pi / 4.0, pi / 4.0)
        return action
    
