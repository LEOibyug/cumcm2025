import numpy as np
import math
import matplotlib.pyplot as plt
import Items
from mpl_toolkits.mplot3d import Axes3D # 确保导入 Axes3D
from matplotlib.animation import FuncAnimation # 导入 FuncAnimation

# 从 utils.py 导入功能函数
from utils import plot_participants

fake_target = Items.Plot(np.array([0, 0, 0]))
target = Items.Volume(np.array([0, 200, 5]), 7, 10)

m1 = Items.Missile(np.array([20000, 0, 2000]), fake_target)

fy_dir_1 = np.array([17800,0,1800]) - fake_target.pos
fy_speed_1 = 120
drop_t1 = [1.5] # 假设在 1.5s 时投掷烟雾
fy1 = Items.Drone(np.array([17800,0,1800]), fy_dir_1, fy_speed_1)

# 仿真参数
TIME_STEP = 0.1
SIMULATION_DURATION = 1000
ANIMATION_INTERVAL_MS = 50 # 动画更新间隔，单位毫秒 (例如 50ms = 20帧/秒)

participant = [fake_target,m1,fy1]
for samples in target.get_sample():
    participant.append(samples)

# 设置绘图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 存储历史位置用于绘制轨迹
history_pos = {p: [] for p in participant}

# --- 仿真状态管理 ---
current_time = 0.0
# 用于跟踪烟雾是否已投掷，避免重复添加
smoke_dropped_times = set() 

def update_simulation_state():

    global current_time, participant, history_pos, smoke_dropped_times

    # 检查是否需要投掷烟雾
    for drop_time in drop_t1:
        # 使用一个小的容差进行浮点数比较
        if abs(current_time - drop_time) < TIME_STEP / 2 and drop_time not in smoke_dropped_times:
            smoke = fy1.drop(3.6) # 假设 drop 方法返回一个 Items.Smoke 实例
            participant.append(smoke)
            history_pos[smoke] = [] # 为新加入的 smoke 实例初始化 history_pos
            smoke_dropped_times.add(drop_time) # 记录已投掷时间

    # 更新所有参与者的位置
    for p in participant:
        p.update(TIME_STEP)
    
    current_time += TIME_STEP

    # 返回当前时间，以便 plot_participants 使用
    return current_time

def animate(frame):
    """
    FuncAnimation 的回调函数，每帧调用一次。
    负责更新仿真状态和绘图。
    """
    global current_time

    # 推进仿真状态
    current_sim_time = update_simulation_state()

    # 调用 plot_participants 进行绘图更新
    plot_participants(current_sim_time, participant, history_pos, fake_target, ax)
    
    # 如果达到仿真结束时间，停止动画
    if current_sim_time >= SIMULATION_DURATION:
        ani.event_source.stop()
        print(f"Simulation finished at {current_sim_time:.2f}s")

# 创建动画
# blit=False 通常在 3D 绘图中更稳定
ani = FuncAnimation(fig, animate, interval=ANIMATION_INTERVAL_MS, blit=False) 

plt.show()

