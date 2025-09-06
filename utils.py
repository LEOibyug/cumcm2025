# utils.py

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
from tqdm import tqdm 
import Items


def plot_sphere(center=(0, 0, 0), radius=1.0, ax=None, **kwargs):
    """
    使用 Matplotlib 绘制一个球体。

    参数:
    center (tuple): 球体的中心坐标 (x, y, z)。默认为 (0, 0, 0)。
    radius (float): 球体的半径。默认为 1.0。
    ax (matplotlib.axes.Axes3D): 可选的 Axes3D 对象。如果未提供，将创建一个新的。
    **kwargs: 传递给 ax.plot_surface 的额外参数，例如 color, alpha, cmap 等。
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # 减少点数以提高性能
    phi = np.linspace(0, np.pi, 15)
    theta = np.linspace(0, 2 * np.pi, 15)

    phi, theta = np.meshgrid(phi, theta)

    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)

    return ax.plot_surface(x, y, z, **kwargs)


def is_line_segment_obscured_by_sphere(p1, p2, sphere_center, sphere_radius):
    line_vec = p2 - p1
    p1_to_center_vec = sphere_center - p1

    line_vec_len_sq = np.dot(line_vec, line_vec)

    if line_vec_len_sq == 0:
        return np.linalg.norm(p1 - sphere_center) <= sphere_radius


    t = np.dot(p1_to_center_vec, line_vec) / line_vec_len_sq
    
    if t < 0.0:
        closest_point_on_line = p1
    elif t > 1.0:
        closest_point_on_line = p2
    else:
        closest_point_on_line = p1 + t * line_vec

    dist_closest_to_center = np.linalg.norm(closest_point_on_line - sphere_center)

    return dist_closest_to_center <= sphere_radius

def run_simulation(
    missile_list,                 # 导弹列表
    drone_list,                   # 无人机列表
    static_participants_list,     # 静态参与者列表 (包含 fake_target 和 target 采样点)
    drop_events,                  # 投掷事件字典
    time_step,                    # 仿真步长
    simulation_duration,          # 仿真总时长
    smoke_radius,                  # 烟雾球体半径
    exist_smoke = []
):
    """
    drop_events (dict): 投掷事件字典，键为投掷时间(float)，值为列表，
                        列表中每个元素为 (无人机编号(int), 烟雾倒计时(float))。
                        例如: {1.5: [(1, 3.6)], 5.0: [(2, 2.0), (1, 4.0)]}
    返回:
    dict: 包含遮挡时长的字典。
          键为导弹ID (int)，值为该导弹的遮挡总时长。
          特殊键 'all_missiles_obscured' (str)，值为所有导弹同时被遮挡的总时长。
          例如: {1: 10.5, 2: 8.2, 'all_missiles_obscured': 3.1}
    """
    # 构建导弹和无人机的ID映射
    print("----------------------------------------------------------------------")
    print(f"Simulation parameters: time_step={time_step}, simulation_duration={simulation_duration}, smoke_radius={smoke_radius}")
    for i in drone_list:
        print(f"Drone {i.id} with speed {i.speed} and direction {i.direction}")
    print(f"Drone drop events: {drop_events}")
    print("----------------------------------------------------------------------")
    missiles_by_id = {}
    for m in missile_list:
        missiles_by_id[m.id] = m
    drones_by_id = {}
    for d in drone_list:
        drones_by_id[d.id] = d

    initial_dynamic_participants = []
    initial_dynamic_participants.extend(exist_smoke)
    initial_dynamic_participants.extend(missile_list)
    initial_dynamic_participants.extend(drone_list)
    
    current_simulation_dynamic_participants = list(initial_dynamic_participants)
    # 提取用于遮挡检测的静态目标点
    target_sample_points = [item for item in static_participants_list if isinstance(item, Items.Plot)]
    num_target_lines_to_check = len(target_sample_points)
    # 初始化遮挡时长
    # 对于每个导弹，以及一个用于所有导弹同时被遮挡的累加器
    obscured_durations = {m_id: 0.0 for m_id in missiles_by_id.keys()}
    obscured_durations['all_missiles_obscured'] = 0.0
    is_obscured = {m_id: False for m_id in missiles_by_id.keys()}
    current_time_pre_calc = 0.0
    
    # 使用一个集合来跟踪已经触发的投弹事件，避免重复投弹
    smoke_dropped_event_keys = set() 
    
    num_frames = int(simulation_duration / time_step)

    sorted_drop_events = sorted(drop_events.items())
    drop_event_idx = 0 # 用于跟踪当前要检查的投弹事件索引

    for frame_idx in range(num_frames + 1):
        
        # 检查是否需要投掷烟雾
        while drop_event_idx < len(sorted_drop_events):
            drop_time_event, events_at_this_time = sorted_drop_events[drop_event_idx]
            
            if current_time_pre_calc >= drop_time_event - time_step / 2 and \
               current_time_pre_calc < drop_time_event + time_step / 2: 
                for drone_id, clock_value in events_at_this_time:
                    event_key = (drop_time_event, drone_id, clock_value) 
                    if event_key not in smoke_dropped_event_keys:
                        drone_to_drop = drones_by_id.get(drone_id)
                        if drone_to_drop:
                            smoke = drone_to_drop.drop(clock_value)
                            current_simulation_dynamic_participants.append(smoke)
                            smoke_dropped_event_keys.add(event_key)
                            print(f"Smoke dropped by drone {drone_id} at {current_time_pre_calc}s.")
                        else:
                            print(f"Warning: Drone with ID {drone_id} not found for drop event at time {drop_time_event}s.")
                drop_event_idx += 1 # 处理完这个时间点的所有事件，移动到下一个
            elif current_time_pre_calc > drop_time_event + time_step / 2:
   
                drop_event_idx += 1
            else:
                break


        # 更新所有动态参与者的位置
        for p in current_simulation_dynamic_participants:
            p.update(time_step)
        
        active_smokes = [s for s in current_simulation_dynamic_participants if isinstance(s, Items.Smoke) and s.display]
        # 跟踪每个导弹的遮挡
        missiles_obscured_status = {} 
        
        # 遍历所有导弹，进行单独的遮挡检测
        for missile_id, missile_obj in missiles_by_id.items():
            missile_pos = missile_obj.pos
            
            current_missile_obscured_line_count = 0
            
            if active_smokes and target_sample_points:
                for target_sample in target_sample_points:
                    line_p1 = target_sample.pos
                    line_p2 = missile_pos 
                    
                    line_is_obscured_by_any_smoke = False
                    for smoke_obj in active_smokes:
                        if is_line_segment_obscured_by_sphere(line_p1, line_p2, smoke_obj.pos, smoke_radius):
                            line_is_obscured_by_any_smoke = True
                            break 
                    
                    if line_is_obscured_by_any_smoke:
                        current_missile_obscured_line_count += 1
            
            # 判断是否被完全遮挡
            is_this_missile_fully_obscured = (num_target_lines_to_check > 0) and \
                                              (current_missile_obscured_line_count == num_target_lines_to_check)
            
            missiles_obscured_status[missile_id] = is_this_missile_fully_obscured
            
            if is_this_missile_fully_obscured:
                if not is_obscured[missile_id] :
                    print(f"\nMissile {missile_id} is fully obscured at time {current_time_pre_calc}s")
                    is_obscured[missile_id] = True
                obscured_durations[missile_id] += time_step
            elif is_obscured[missile_id] :
                print(f"Missile {missile_id} is no longer fully obscured at time {current_time_pre_calc}s")
                is_obscured[missile_id] = False
        
        # 判断所有导弹是否同时被遮挡
        all_missiles_simultaneously_obscured = False
        if missiles_by_id: 
            all_missiles_simultaneously_obscured = all(status for status in missiles_obscured_status.values())
        
        if all_missiles_simultaneously_obscured:
            obscured_durations['all_missiles_obscured'] += time_step
        
        current_time_pre_calc += time_step
    
    print("Simulation complete. Obscured Durations:")
    for key, value in obscured_durations.items():
        if key == 'all_missiles_obscured':
            print(f"  All Missiles Obscured: {value:.4f}s")
        else:
            print(f"  Missile {key} Obscured: {value:.4f}s")
    print("---------------------------------------------\n")
    return obscured_durations

