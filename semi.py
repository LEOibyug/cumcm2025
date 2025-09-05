import numpy as np
import math
import matplotlib.pyplot as plt
import Items
import utils
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# 从 utils.py 导入功能函数
from utils import plot_sphere, is_line_segment_obscured_by_sphere

fake_target = Items.Plot(np.array([0, 0, 0]))
target = Items.Volume(np.array([0, 200, 5]), 7, 10)
stastic_participants_list = [fake_target]
for i in target.get_sample():
    stastic_participants_list.append(i)

m1 = Items.Missile(np.array([20000, 0, 2000]), fake_target,1)
# m2 = Items.Missile(np.array([19000, 600, 2100]), fake_target)
# m3 = Items.Missile(np.array([18000,-600,1900]), fake_target)

fy_dir_1 = [-0.99993885,0.01105904,0]
fy_speed_1 = 120
fy1 = Items.Drone(np.array([17800,0,1800]), fy_dir_1, fy_speed_1,1)

# fy_dir_2 = np.array([0, 0, 1])
# fy_speed_2 = 100
# drop_t2 = [10]
# fy2 = Items.Drone(np.array([12000,1400,1400]), fy_dir_2, fy_speed_2)

# fy_dir_3 = np.array([0, 0, 1])
# fy_speed_3 = 100
# drop_t2 = [10]
# fy3 = Items.Drone(np.array([6000, -3000, 700]), fy_dir_3, fy_speed_3)

# fy_dir_4 = np.array([0, 0, 1])
# fy_speed_4 = 100
# drop_t2 = [10]
# fy4 = Items.Drone(np.array([11000, 2000, 1800]), fy_dir_4, fy_speed_4)

# fy_dir_5 = np.array([0, 0, 1])
# fy_speed_5 = 100
# drop_t2 = [10]
# fy5 = Items.Drone(np.array([13000, -2000, 1300]), fy_dir_5, fy_speed_5)


# 仿真参数
TIME_STEP = 0.01
SIMULATION_DURATION = 20
SMOKE_RADIUS = 10 # 烟雾球体半径
drop_events = {
    0.9758:[(1,3.8205)],
    2.4758:[(1,5)],
    3.9758:[(1,5)],
}

res = utils.run_simulation([m1],[fy1],stastic_participants_list,drop_events,TIME_STEP,SIMULATION_DURATION,SMOKE_RADIUS)