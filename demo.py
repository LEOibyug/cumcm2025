import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Items # 确保 Items.py 文件在当前目录下
import random # 导入 random 模块用于随机选择

# --- 1. 创建 Volume 对象和获取 samples ---
v = Items.Volume([0,0,5],5,10)
samples = v.get_sample()

# 提取所有样本点的坐标
x_coords = [s.pos[0] for s in samples]
y_coords = [s.pos[1] for s in samples]
z_coords = [s.pos[2] for s in samples]

# --- 2. 创建 Missile 对象 ---
missile_pos = [20, 20, 20]
temp_target_for_missile = Items.Item([0,0,0])
missile = Items.Missile(missile_pos, temp_target_for_missile, id=1)

# --- 3. 随机选择5个样本点 ---
if len(samples) < 5:
    print("Warning: Not enough samples to pick 5 random points. Picking all available samples.")
    random_sample_points = samples
else:
    random_sample_points = random.sample(samples, 5)

# 提取随机选择点的坐标
random_x = [s.pos[0] for s in random_sample_points]
random_y = [s.pos[1] for s in random_sample_points]
random_z = [s.pos[2] for s in random_sample_points]

# --- 4. 定义球体参数并生成点 ---
sphere_center = [10, 10, 10]
sphere_radius = 5

# 生成球体表面的点
# 创建网格
u = np.linspace(0, 2 * np.pi, 50) # 方位角 phi
v = np.linspace(0, np.pi, 25)     # 极角 theta

# 计算球体坐标
sphere_x = sphere_center[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
sphere_y = sphere_center[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
sphere_z = sphere_center[2] + sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))


# --- 5. 绘图部分 ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制 Volume 的采样点 (蓝色)
ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o', s=5, label='Volume Samples')

# 绘制 Missile 对象 (红色星号)
ax.scatter(missile.pos[0], missile.pos[1], missile.pos[2], c='r', marker='*', s=200, label='Missile')

# 绘制随机选择的5个样本点 (绿色大圆圈)
ax.scatter(random_x, random_y, random_z, c='g', marker='o', s=100)

# 绘制 Missile 与随机选择点的连线
for i, sample_point in enumerate(random_sample_points):
    m_x, m_y, m_z = missile.pos
    s_x, s_y, s_z = sample_point.pos
    ax.plot([m_x, s_x], [m_y, s_y], [m_z, s_z], c='purple', linestyle='--', linewidth=1, alpha=0.7, label="Missile's Vision" if i == 0 else "")

# 绘制球体 (使用 plot_surface)
ax.plot_surface(sphere_x, sphere_y, sphere_z, color='c', alpha=1, label='Smoke') # 'c' for cyan



# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')



# 设置X轴范围
ax.set_xlim([-10, 30])
# 设置Y轴范围
ax.set_ylim([-10, 30])
# 设置Z轴范围
ax.set_zlim([-1, 25])

# 添加图例
ax.legend()

# 显示图形
plt.show()
