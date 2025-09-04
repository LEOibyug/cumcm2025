# utils.py

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # 确保导入 Axes3D

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

    # 减少点数以提高性能，如果需要更平滑可以增加
    phi = np.linspace(0, np.pi, 50)
    theta = np.linspace(0, 2 * np.pi, 50)

    phi, theta = np.meshgrid(phi, theta)

    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)

    # 返回 Poly3DCollection 对象，以便可以用于图例
    return ax.plot_surface(x, y, z, **kwargs)

def plot_participants(current_time, participant, history_pos, fake_target, ax):
    """
    绘制所有参与者及其轨迹。

    参数:
    current_time (float): 当前仿真时间。
    participant (list): 包含所有参与者对象的列表。
    history_pos (dict): 存储每个参与者历史位置的字典。
    fake_target (Items.Plot): 模拟的假目标实例。
    ax (matplotlib.axes.Axes3D): 用于绘图的 Axes3D 对象。
    """
    ax.clear()  
    ax.set_title(f"Simulation Time: {current_time:.2f}s")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    all_x = []
    all_y = []
    all_z = []
    plotted_labels = set()
    sphere_handle = None 
    import Items

    for p in participant:
        if not p.display: 
            continue
        
        x, y, z = p.pos
        all_x.append(x)
        all_y.append(y)
        all_z.append(z)
        
        # 确保 history_pos 中有 p 的键
        if p not in history_pos:
            history_pos[p] = []
        history_pos[p].append(p.pos.copy()) # 记录当前位置

        label_to_add = None
        if isinstance(p, Items.Missile):
            if 'Missile' not in plotted_labels:
                label_to_add = 'Missile'
                plotted_labels.add('Missile')
            ax.scatter(x, y, z, color='red', marker='^', s=50, label=label_to_add)
        elif isinstance(p, Items.Drone):
            if 'Drone' not in plotted_labels:
                label_to_add = 'Drone'
                plotted_labels.add('Drone')
            ax.scatter(x, y, z, color='green', marker='o', s=50, label=label_to_add)
        elif isinstance(p, Items.Plot):
            if p is fake_target:
                if 'Fake Target' not in plotted_labels:
                    label_to_add = 'Fake Target'
                    plotted_labels.add('Fake Target')
                ax.scatter(x, y, z, color='blue', marker='x', s=100, label=label_to_add)
            else:
                if 'Target Sample' not in plotted_labels:
                    label_to_add = 'Target Sample'
                    plotted_labels.add('Target Sample')
                ax.scatter(x, y, z, color='purple', marker='.', s=20, label=label_to_add)
        elif isinstance(p, Items.Smoke):
            # 绘制球体
            if 'Smoke Sphere' not in plotted_labels:
                sphere_handle = plot_sphere(center=p.pos, radius=10, ax=ax, color='gray', alpha=0.3)
                plotted_labels.add('Smoke Sphere')
            else:
                plot_sphere(center=p.pos, radius=10, ax=ax, color='gray', alpha=0.3)
            


        # 绘制轨迹
        if len(history_pos[p]) > 1:
            hist_arr = np.array(history_pos[p])
            ax.plot(hist_arr[:, 0], hist_arr[:, 1], hist_arr[:, 2], linestyle='--', alpha=0.5, color='gray')

    # 设置坐标轴范围
    if all_x:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        min_z, max_z = min(all_z), max(all_z)
        

        if 'Smoke Sphere' in plotted_labels:
            min_x -= 10
            max_x += 10
            min_y -= 10
            max_y += 10
            min_z -= 10
            max_z += 10

        margin_factor = 0.1
        x_range = max_x - min_x
        y_range = max_y - min_y
        z_range = max_z - min_z
        
        if x_range == 0: x_range = 10
        if y_range == 0: y_range = 10
        if z_range == 0: z_range = 10
        
        max_overall_range = max(x_range, y_range, z_range)
        
        ax.set_xlim(min_x - max_overall_range * margin_factor, max_x + max_overall_range * margin_factor)
        ax.set_ylim(min_y - max_overall_range * margin_factor, max_y + max_overall_range * margin_factor)
        ax.set_zlim(min_z - max_overall_range * margin_factor, max_z + max_overall_range * margin_factor)


    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    if sphere_handle is not None and 'Smoke Sphere' in plotted_labels:
        if 'Smoke Sphere' not in by_label:
            by_label['Smoke Sphere'] = sphere_handle

    ax.legend(by_label.values(), by_label.keys())

