import matplotlib.pyplot as plt

log_file_name = 'optimization_log_20250905_161856.log' 

# 定义所有要绘制的数值列
plot_columns = [
    'fy_speed_1', 
    'fy_dir_x', 
    'fy_dir_y', 
    'drop_time', 
    'clock_value', 
    'Obscured_Duration'
]

# 存储数据的字典，键为列名，值为对应的数据列表
data_dict = {col: [] for col in plot_columns}
iterations_counter = [] # 用于表示行号或数据点的顺序

try:
    with open(log_file_name, 'r') as f:
        header = f.readline().strip().split(',')
        
        # 找到所有需要绘制的列的索引
        col_indices = {}
        try:
            for col_name in plot_columns:
                col_indices[col_name] = header.index(col_name)
        except ValueError as e:
            print(f"Error: Missing expected column in header: {e}")
            print(f"Header found: {header}")
            exit()

        line_num = 0 
        for line in f:
            stripped_line = line.strip()
            parts = stripped_line.split(',')
            
            if len(parts) == len(header) and parts[0] == 'Iteration':
                try:
                    for col_name in plot_columns:
                        data_dict[col_name].append(float(parts[col_indices[col_name]]))
                    
                    line_num += 1
                    iterations_counter.append(line_num)

                except ValueError:
                    print(f"Warning: Skipping malformed data in line: '{stripped_line}' (ValueError during float conversion)")
                    continue
            else:
                print(f"Warning: Skipping incomplete, empty, or malformed line: '{stripped_line}' (Column count mismatch or missing 'Iteration' tag)")
                continue

except FileNotFoundError:
    print(f"Error: The file '{log_file_name}' was not found. Please ensure it exists in the same directory as the script.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()

# 检查是否成功读取到数据
if not iterations_counter:
    print("No valid data was successfully read from the file. Please check the file content and format.")
    exit()
else:
    print(f"Successfully read {len(iterations_counter)} data points.")


# 绘制散点图，使用子图
num_plots = len(plot_columns)
num_rows = (num_plots + 1) // 2 # 计算行数，向上取整
num_cols = 2 # 固定为2列

fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4), sharex=True) # 调整figsize
axs = axs.flatten() # 将axs展平，方便通过索引访问

# 遍历所有要绘制的列，并在对应的子图中绘制
for i, col_name in enumerate(plot_columns):
    ax = axs[i] # 获取当前子图的Axes对象
    # 将 linestyle 设置为 'None'，只显示标记，不连接点
    ax.plot(iterations_counter, data_dict[col_name], label=col_name, marker='o', linestyle='None', markersize=4, alpha=0.7) # alpha设置透明度，点多时避免重叠
    ax.set_ylabel(f'{col_name} Value')
    ax.set_title(f'{col_name} Over Iterations')
    ax.grid(True)
    ax.legend()

# 为最底部的子图设置X轴标签
# 检查num_plots是奇数还是偶数，决定哪个子图是最后一行的
if num_plots % num_cols != 0: # 如果不是偶数列，最后一行的最后一个子图可能没有数据
    for i in range(num_plots, num_rows * num_cols):
        fig.delaxes(axs[i]) # 删除多余的空子图

# 确保最底部的子图有X轴标签
# 找到最后一行的所有子图，并为它们设置X轴标签
for i in range(num_plots - num_cols, num_plots):
    if i >= 0 and i < len(axs): # 确保索引有效且在axs范围内
        axs[i].set_xlabel('Data Point Index (Iteration)')


plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局，为总标题留出空间
fig.suptitle(f'Log Data Trends from {log_file_name} (Scatter Plot)', fontsize=16) # 添加总标题，并注明是散点图

plt.show()
