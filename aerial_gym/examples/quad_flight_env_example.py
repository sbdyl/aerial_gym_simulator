from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args

def find_asset_indices_by_type(env_manager, asset_type):
    """根据asset类型查找索引"""
    asset_dicts = env_manager.global_asset_dicts[0]
    indices = []
    for i, asset_dict in enumerate(asset_dicts):
        if asset_dict['asset_type'] == asset_type:
            indices.append(i)
    return indices

if __name__ == "__main__":
    logger.warning(
        "\n\n\nWhile possible, a dynamic environment will slow down the simulation by a lot. Use with caution. Native Isaac Gym cameras work faster than Warp in this case.\n\n\n"
    )
    args = get_args()
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="quad_flight_dynamic_env",
        robot_name="lmf2",
        controller_name="lmf2_position_control",
        args=None,
        device="cuda:0",
        num_envs=args.num_envs,
        headless=args.headless,
        use_warp=args.use_warp,
    )
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    env_manager.reset()
    num_assets_in_env = (
        env_manager.IGE_env.num_assets_per_env - 1
    )  # subtract 1 because the robot is also an asset
    print(f"Number of assets in the environment: {num_assets_in_env}")
    num_envs = env_manager.num_envs

    uav_indices = find_asset_indices_by_type(env_manager, 'dynamic_uav')
    print(f"dynamic_uav indices: {uav_indices}")

    asset_twist = torch.zeros((env_manager.num_envs, num_assets_in_env, 6), device="cuda:0")
    
    for i in range(10000):
        # 只控制第一个panel做圆形运动
        asset_twist[:, uav_indices[0], 0] = torch.sin(0.1 * i * torch.ones_like(asset_twist[:, uav_indices[0], 0]))
        asset_twist[:, uav_indices[0], 1] = torch.cos(0.1 * i * torch.ones_like(asset_twist[:, uav_indices[0], 1]))
        
        env_manager.step(actions=actions, env_actions=asset_twist)
