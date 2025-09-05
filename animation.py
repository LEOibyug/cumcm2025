import numpy as np
import math
import matplotlib.pyplot as plt
import Items
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# 从 utils.py 导入功能函数
from utils import plot_sphere, is_line_segment_obscured_by_sphere

fake_target = Items.Plot(np.array([0, 0, 0]))
target = Items.Volume(np.array([0, 200, 5]), 7, 10)

m1 = Items.Missile(np.array([20000, 0, 2000]), fake_target,1)

fy_dir_1 = [0.9956142,0.0935541,0]
fy_speed_1 = 103.6163
drop_t1 = [0.8725] # 假设在 1.5s 时投掷烟雾
fy1 = Items.Drone(np.array([17800,0,1800]), fy_dir_1, fy_speed_1,1)

# 仿真参数
TIME_STEP = 0.005
SIMULATION_DURATION = 20
ANIMATION_INTERVAL_MS = 10 # 动画更新间隔ms
SMOKE_RADIUS = 10 # 烟雾球体半径

# 将参与者分为动态和静态两类
dynamic_initial_participants = [fake_target, m1, fy1]
static_target_samples = []
for samples in target.get_sample():
    static_target_samples.append(samples)

# --- 预计算所有仿真帧的数据 ---
print("Pre-calculating simulation data...")
all_frames_data = [] # 存储每一帧的所有动态参与者的位置和状态
all_history_pos = {p: [] for p in dynamic_initial_participants} # 存储每个动态参与者的完整历史轨迹

# 复制 dynamic_initial_participants，因为在模拟过程中会添加新的 smoke 对象
current_simulation_dynamic_participants = list(dynamic_initial_participants)

current_time_pre_calc = 0.0
smoke_dropped_times_pre_calc = set()

# 遮挡时间跟踪变量
current_obscured_duration = 0.0
max_obscured_duration = 0.0
is_currently_obscured = False # 用于判断所有线是否都被遮挡

# 计算总帧数
num_frames = int(SIMULATION_DURATION / TIME_STEP)

for frame_idx in tqdm(range(num_frames + 1), desc="Simulating Frames"):
    
    # 检查是否需要投掷烟雾
    for drop_time in drop_t1:
        if abs(current_time_pre_calc - drop_time) < TIME_STEP / 2 and drop_time not in smoke_dropped_times_pre_calc:
            smoke = fy1.drop(0.2591)
            current_simulation_dynamic_participants.append(smoke)
            all_history_pos[smoke] = [] # 为新加入的 smoke 实例初始化历史位置
            smoke_dropped_times_pre_calc.add(drop_time)

    # 更新所有动态参与者的位置
    for p in current_simulation_dynamic_participants:
        p.update(TIME_STEP)
    
    # --- 遮挡检测逻辑 ---
    missile_pos = m1.pos # 获取当前导弹位置
    active_smokes = [s for s in current_simulation_dynamic_participants if isinstance(s, Items.Smoke) and s.display]

    obscured_line_count = 0 # 统计被遮挡的直线数量
    
    if active_smokes: # 只有当有活动的烟雾时才进行遮挡检测
        for target_sample in static_target_samples:
            line_p1 = target_sample.pos
            line_p2 = missile_pos
            
            line_is_obscured_by_any_smoke = False
            for smoke_obj in active_smokes:
                if is_line_segment_obscured_by_sphere(line_p1, line_p2, smoke_obj.pos, SMOKE_RADIUS):
                    line_is_obscured_by_any_smoke = True
                    break # 这条线被某个烟雾遮挡了，检查下一条线
            
            if line_is_obscured_by_any_smoke:
                obscured_line_count += 1
    
    # 判断所有线是否都被遮挡，用于计时
    all_lines_obscured_for_timing = (obscured_line_count == len(static_target_samples)) and (len(static_target_samples) > 0)

    # 更新遮挡时间
    if all_lines_obscured_for_timing:
        if not is_currently_obscured:
            is_currently_obscured = True
            current_obscured_duration = 0.0 # 开始新的计时
        current_obscured_duration += TIME_STEP
        max_obscured_duration = max(max_obscured_duration, current_obscured_duration)
    else:
        if is_currently_obscured:
            is_currently_obscured = False
            # current_obscured_duration 保持其值，直到下一次进入遮挡状态才清零
            # max_obscured_duration 已经更新过了
    
    # 记录当前帧的所有动态参与者的状态
    current_frame_data = {
        'time': current_time_pre_calc,
        'dynamic_participants_state': [],
        'obscured_duration': current_obscured_duration,
        'obscured_line_count': obscured_line_count # 新增：被遮挡的直线数量
    }
    for p in current_simulation_dynamic_participants:
        current_frame_data['dynamic_participants_state'].append({
            'instance': p,
            'pos': p.pos.copy(),
            'display': p.display
        })
        all_history_pos[p].append(p.pos.copy())

    all_frames_data.append(current_frame_data)
    
    current_time_pre_calc += TIME_STEP

print(f"Pre-calculation complete. Total frames: {len(all_frames_data)}")
print(f"Max continuous obscured duration: {max_obscured_duration:.6f}s")


# --- 设置绘图 ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# --- 初始化图形对象 ---
dynamic_plot_objects = {}
dynamic_trajectory_lines = {}
legend_handles = {}
legend_labels = {}

colors = {
    Items.Missile: 'red', Items.Drone: 'green', Items.Plot: 'blue', Items.Smoke: 'gray'
}
markers = {
    Items.Missile: '^', Items.Drone: 'o', Items.Plot: 'x', Items.Smoke: 'o' 
}
sizes = {
    Items.Missile: 50, Items.Drone: 50, Items.Plot: 100, Items.Smoke: 100
}

# 1. 绘制静态目标采样点 (只绘制一次)
print("Plotting static target samples...")
target_sample_x = [p.pos[0] for p in static_target_samples]
target_sample_y = [p.pos[1] for p in static_target_samples]
target_sample_z = [p.pos[2] for p in static_target_samples]
if target_sample_x:
    static_plot_obj = ax.scatter(target_sample_x, target_sample_y, target_sample_z, color='purple', marker='.', s=20, label='Target Sample')
    legend_handles['Target Sample'] = static_plot_obj
    legend_labels['Target Sample'] = 'Target Sample'

# 2. 初始化动态参与者的图形对象
for p_instance in all_history_pos.keys():
    if isinstance(p_instance, Items.Missile):
        dynamic_plot_objects[p_instance], = ax.plot([], [], [], color=colors[type(p_instance)], marker=markers[type(p_instance)], markersize=math.sqrt(sizes[type(p_instance)]))
        if 'Missile' not in legend_labels:
            legend_handles['Missile'] = dynamic_plot_objects[p_instance]
            legend_labels['Missile'] = 'Missile'
    elif isinstance(p_instance, Items.Drone):
        dynamic_plot_objects[p_instance], = ax.plot([], [], [], color=colors[type(p_instance)], marker=markers[type(p_instance)], markersize=math.sqrt(sizes[type(p_instance)]))
        if 'Drone' not in legend_labels:
            legend_handles['Drone'] = dynamic_plot_objects[p_instance]
            legend_labels['Drone'] = 'Drone'
    elif isinstance(p_instance, Items.Plot): # This is fake_target
        dynamic_plot_objects[p_instance], = ax.plot([], [], [], color='blue', marker='x', markersize=math.sqrt(sizes[type(p_instance)]))
        if 'Fake Target' not in legend_labels:
            legend_handles['Fake Target'] = dynamic_plot_objects[p_instance]
            legend_labels['Fake Target'] = 'Fake Target'
    elif isinstance(p_instance, Items.Smoke):
        dynamic_plot_objects[p_instance], = ax.plot([], [], [], color=colors[type(p_instance)], marker='o', markersize=math.sqrt(sizes[type(p_instance)]), alpha=0.5)
        if 'Smoke' not in legend_labels:
            legend_handles['Smoke'] = dynamic_plot_objects[p_instance]
            legend_labels['Smoke'] = 'Smoke'
    
    dynamic_trajectory_lines[p_instance], = ax.plot([], [], [], linestyle='--', alpha=0.5, color='gray')

# 3. 初始化经过原点的坐标轴
axis_length = 25000 
x_axis_line, = ax.plot([0, axis_length], [0, 0], [0, 0], color='black', linestyle='-', linewidth=1)
y_axis_line, = ax.plot([0, 0], [0, axis_length], [0, 0], color='black', linestyle='-', linewidth=1)
z_axis_line, = ax.plot([0, 0], [0, 0], [0, axis_length], color='black', linestyle='-', linewidth=1)

# 4. 初始化文本显示对象
obscured_duration_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, color='red', fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
obscured_count_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, color='blue', fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))


# 初始设置轴标签和标题
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Simulation Time: 0.00s")

# 初始创建图例
ax.legend(legend_handles.values(), legend_labels.values(), loc='best')

# --- 动画函数 ---
def animate(frame_idx):
    if frame_idx >= len(all_frames_data):
        print("Animation finished.")
        ani.event_source.stop()
        return []

    frame_data = all_frames_data[frame_idx]
    current_time = frame_data['time']
    current_obscured_duration_display = frame_data['obscured_duration']
    current_obscured_line_count_display = frame_data['obscured_line_count'] # 获取被遮挡的直线数量
    
    ax.set_title(f"Simulation Time: {current_time:.4f}s")
    obscured_duration_text.set_text(f"Obscured Duration: {current_obscured_duration_display:.4f}s")
    obscured_count_text.set_text(f"Obscured Lines: {current_obscured_line_count_display}/{len(static_target_samples)}") # 更新文本

    all_x = []
    all_y = []
    all_z = []
    
    all_x.extend(target_sample_x)
    all_y.extend(target_sample_y)
    all_z.extend(target_sample_z)

    updated_artists = [obscured_duration_text, obscured_count_text, ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]

    for p_state in frame_data['dynamic_participants_state']:
        p_instance = p_state['instance']
        p_pos = p_state['pos']
        p_display = p_state['display']

        if p_instance in dynamic_plot_objects:
            plot_obj = dynamic_plot_objects[p_instance]
            plot_obj.set_data_3d([p_pos[0]], [p_pos[1]], [p_pos[2]])
            plot_obj.set_visible(p_display)
            updated_artists.append(plot_obj)
            
        if p_instance in dynamic_trajectory_lines:
            line_obj = dynamic_trajectory_lines[p_instance]
            hist_arr = np.array(all_history_pos[p_instance][:frame_idx+1])
            if len(hist_arr) > 0:
                line_obj.set_data_3d(hist_arr[:, 0], hist_arr[:, 1], hist_arr[:, 2])
                line_obj.set_visible(p_display)
                updated_artists.append(line_obj)
            else:
                line_obj.set_visible(False)
                updated_artists.append(line_obj)

        if p_display:
            all_x.append(p_pos[0])
            all_y.append(p_pos[1])
            all_z.append(p_pos[2])
            if isinstance(p_instance, Items.Smoke):
                size = sizes[Items.Smoke] / 2
                all_x.extend([p_pos[0] - size, p_pos[0] + size])
                all_y.extend([p_pos[1] - size, p_pos[1] + size])
                all_z.extend([p_pos[2] - size, p_pos[2] + size])

    if all_x:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        min_z, max_z = min(all_z), max(all_z)
        
        margin_factor = 0.1
        x_range = max_x - min_x
        y_range = max_y - min_y
        z_range = max_z - min_z
        
        if x_range == 0: x_range = 10
        if y_range == 0: y_range = 10
        if z_range == 0: z_range = 10
        
        max_overall_range = max(x_range, y_range, z_range)
        
        ax.set_xlim(min(0, min_x - max_overall_range * margin_factor), max(axis_length, max_x + max_overall_range * margin_factor))
        ax.set_ylim(min(0, min_y - max_overall_range * margin_factor), max(axis_length, max_y + max_overall_range * margin_factor))
        ax.set_zlim(min(0, min_z - max_overall_range * margin_factor), max(axis_length, max_z + max_overall_range * margin_factor))
    else:
        ax.set_xlim(0, axis_length)
        ax.set_ylim(0, axis_length)
        ax.set_zlim(0, axis_length)

    updated_artists.extend([x_axis_line, y_axis_line, z_axis_line])

    return updated_artists


ani = FuncAnimation(fig, animate, frames=len(all_frames_data), interval=ANIMATION_INTERVAL_MS, blit=True) 


# with tqdm(total=len(all_frames_data), desc="Saving Animation") as pbar:
#     ani.save("1.mp4", writer='ffmpeg', fps=1000 / ANIMATION_INTERVAL_MS, dpi=150, progress_callback=lambda i, n: pbar.update(1))

plt.show()

