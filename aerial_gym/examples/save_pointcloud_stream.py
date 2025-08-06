import matplotlib.pyplot as plt
import numpy as np
import random
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
from PIL import Image
import matplotlib
import torch
import random
import open3d as o3d

if __name__ == "__main__":
    logger.warning("\n\n\nEnvironment to save a depth/range and segmentation image.\n\n\n")

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # 检测传感器类型
    robot_name = "base_quadrotor_with_depth_lidar"  # 或 "base_quadrotor_with_pointcloud_depth_camera"
    
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="env_with_obstacles",
        robot_name=robot_name,
        controller_name="lee_velocity_control",
        args=None,
        device="cuda:0",
        num_envs=2,
        headless=False,
        use_warp=True,
    )
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    actions[:, 3] = 0.1

    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="实时点云可视化", width=1200, height=800)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    # 设置渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.05, 0.05, 0.05])  # 深色背景
    opt.point_size = 1.5
    
    env_manager.reset()

    # 添加坐标轴
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)  # 较大的坐标轴
    vis.add_geometry(coord_frame)

    # 自适应参数
    first_frame = True
    data_scale = 1.0
    sensor_type = "unknown"

    try:
        frame_count = 0
        for i in range(10000):
            env_manager.step(actions=actions)
            env_manager.render(render_components="sensors")
            env_manager.reset_terminated_and_truncated_envs()

            # 获取点云数据
            pointcloud_data = env_manager.global_tensor_dict["depth_range_pixels"][0, 0].cpu().numpy()
            
            # 数据统计
            height, width = pointcloud_data.shape[:2]
            x_range = [pointcloud_data[:,:,0].min(), pointcloud_data[:,:,0].max()]
            y_range = [pointcloud_data[:,:,1].min(), pointcloud_data[:,:,1].max()]
            z_range = [pointcloud_data[:,:,2].min(), pointcloud_data[:,:,2].max()]
            
            print(f"帧 {frame_count}: 形状: {pointcloud_data.shape}")
            print(f"X: [{x_range[0]:.2f}, {x_range[1]:.2f}], 范围: {x_range[1]-x_range[0]:.2f}")
            print(f"Y: [{y_range[0]:.2f}, {y_range[1]:.2f}], 范围: {y_range[1]-y_range[0]:.2f}")
            print(f"Z: [{z_range[0]:.2f}, {z_range[1]:.2f}], 范围: {z_range[1]-z_range[0]:.2f}")

            # 自动检测传感器类型和数据特征
            if first_frame:
                max_coord_range = max(x_range[1]-x_range[0], y_range[1]-y_range[0], z_range[1]-z_range[0])
                
                if max_coord_range > 100:  # 雷达数据，范围较大
                    sensor_type = "lidar"
                    data_scale = 1.0 # 缩放因子
                    print(f"检测到: 雷达数据，最大范围: {max_coord_range:.1f}m，缩放因子: {data_scale:.2f}")
                    
                    # 雷达过滤策略
                    min_valid_range = 0.5
                    max_valid_range = 15.0
                    
                elif max_coord_range < 10 and min(x_range[0], y_range[0]) > -5:  # 相机数据，范围较小
                    sensor_type = "camera"
                    data_scale = 1.0
                    print(f"检测到: 相机数据，最大范围: {max_coord_range:.1f}m")
                    
                    # 相机过滤策略
                    min_valid_range = 0.1
                    max_valid_range = 15.0
                    
                else:  # 其他情况
                    sensor_type = "other" 
                    data_scale = max_coord_range / 20.0 if max_coord_range > 20 else 1.0
                    print(f"检测到: 其他传感器，最大范围: {max_coord_range:.1f}m，缩放因子: {data_scale:.2f}")
                    
                    min_valid_range = 0.1
                    max_valid_range = max_coord_range * 0.9

            # 重塑为N×3的格式
            points = pointcloud_data.reshape(-1, 3)
            print(f"总点数: {points.shape[0]}")

            # 根据传感器类型应用不同的过滤策略
            if sensor_type == "lidar":
                # 雷达数据过滤：去除极值和明显错误的点
                valid_mask = (
                    (points[:, 0] != -1.0) & (points[:, 1] != -1.0) & (points[:, 2] != -1.0) &  # 去除标记值
                    (np.abs(points[:, 0]) < max_valid_range) &  # X范围限制
                    (np.abs(points[:, 1]) < max_valid_range) &  # Y范围限制
                    (np.abs(points[:, 2]) < max_valid_range) &  # Z范围限制
                    (np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2) > min_valid_range)  # 距离原点最小距离
                )
            elif sensor_type == "camera":
                # 相机数据过滤：主要关注深度值
                valid_mask = (
                    (points[:, 0] != -1.0) & (points[:, 1] != -1.0) & (points[:, 2] != -1.0) &  # 去除标记值
                    (points[:, 2] > min_valid_range) &  # Z > 最小深度
                    (points[:, 2] < max_valid_range) &  # Z < 最大深度
                    (np.abs(points[:, 0]) < max_valid_range) &  # X范围
                    (np.abs(points[:, 1]) < max_valid_range)    # Y范围
                )
            else:
                # 通用过滤
                valid_mask = (
                    (points[:, 0] != -1.0) & (points[:, 1] != -1.0) & (points[:, 2] != -1.0) &
                    (np.abs(points[:, 0]) < max_valid_range) &
                    (np.abs(points[:, 1]) < max_valid_range) &
                    (np.abs(points[:, 2]) < max_valid_range)
                )
            
            valid_points = points[valid_mask]
            print(f"有效点数: {valid_points.shape[0]} / {points.shape[0]} ({100*valid_points.shape[0]/points.shape[0]:.1f}%)")

            if valid_points.shape[0] > 0:
                # 性能优化：下采样
                if valid_points.shape[0] > 15000:
                    indices = np.random.choice(valid_points.shape[0], 15000, replace=False)
                    valid_points = valid_points[indices]
                    print(f"下采样后点数: {valid_points.shape[0]}")

                # 更新点云
                pcd.points = o3d.utility.Vector3dVector(valid_points)
                
                # 基于距离的颜色编码
                if len(valid_points) > 0:
                    # 计算点到原点的距离
                    distances = np.sqrt(np.sum(valid_points**2, axis=1))
                    dist_min, dist_max = distances.min(), distances.max()
                    if dist_max > dist_min:
                        # 使用彩虹色谱：近距离为蓝色，远距离为红色
                        normalized_dist = (distances - dist_min) / (dist_max - dist_min)
                        colors = plt.cm.plasma(normalized_dist)[:, :3]  # plasma色彩更鲜艳
                        pcd.colors = o3d.utility.Vector3dVector(colors)

                # 自适应视角设置
                if first_frame and len(valid_points) > 0:
                    center = valid_points.mean(axis=0)
                    extent = np.max(valid_points, axis=0) - np.min(valid_points, axis=0)
                    max_extent = np.max(extent)
                    
                    print(f"点云中心: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
                    print(f"点云范围: [{extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f}]")
                    print(f"最大范围: {max_extent:.2f}")
                    
                    # 根据数据规模调整视角
                    ctr = vis.get_view_control()
                    
                    if sensor_type == "lidar":
                        # 雷达：从上方和侧方观看
                        ctr.set_front([0.3, -0.3, -0.9])
                        ctr.set_up([0, 0, 1])
                        zoom_factor = 0.01 if max_extent > 100 else 0.05
                    elif sensor_type == "camera":
                        # 相机：从后方观看
                        ctr.set_front([0.5, -0.3, -0.8])
                        ctr.set_up([0, -1, 0])
                        zoom_factor = 0.3
                    else:
                        # 通用视角
                        ctr.set_front([0.4, -0.4, -0.8])
                        ctr.set_up([0, -1, 0])
                        zoom_factor = max(0.01, 10.0 / max_extent)
                    
                    ctr.set_lookat(center)
                    ctr.set_zoom(zoom_factor)
                    print(f"设置缩放因子: {zoom_factor:.3f}")
                    
                    first_frame = False

            # 更新可视化
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            
            frame_count += 1
            
            # 每20帧打印一次统计
            if frame_count % 20 == 0:
                print(f"--- 已处理 {frame_count} 帧 ({sensor_type}) ---\n")

    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    finally:
        vis.destroy_window()
        logger.info("可视化窗口已关闭")