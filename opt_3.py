import numpy as np
import math
import Items
import utils
from tqdm import tqdm
from scipy.optimize import minimize
import datetime

# --- 固定仿真参数 ---
TIME_STEP = 0.01
SIMULATION_DURATION = 40 # 仿真总时长
SMOKE_RADIUS = 10

# --- 静态和固定参与者 ---
fake_target = Items.Plot(np.array([0, 0, 0]))
target = Items.Volume(np.array([0, 200, 5]), 7, 10)

static_target_samples = []
for samples in target.get_sample():
    static_target_samples.append(samples)

# --- 设置日志文件 ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"optimization_log_{timestamp}.log"
log_file = None # 初始化为 None，在 objective 函数中打开

# --- 定义优化目标函数 ---
# 现在接收 9 个参数
def objective_for_minimize(params):
    global log_file # 声明使用全局变量

    if log_file is None: # 第一次调用时打开日志文件
        log_file = open(log_filename, 'w')
        # 写入CSV头部，适应新的参数数量
        log_file.write("Iteration,fy_speed_1,fy_dir_x,fy_dir_y,drop_time_1,clock_value_1,drop_time_2,clock_value_2,drop_time_3,clock_value_3,Obscured_Duration\n")
        # 确保文件被立即刷新，以便在程序崩溃时也能保留部分日志
        log_file.flush()

    # 解包 9 个参数
    fy_speed_1, fy_dir_x, fy_dir_y, \
    drop_time_1, clock_value_1, \
    drop_time_2, clock_value_2, \
    drop_time_3, clock_value_3 = params

    # --- 重新创建导弹实例，确保每次仿真都从初始位置开始 ---
    current_m1 = Items.Missile(np.array([20000, 0, 2000]), fake_target, 1)
    current_missile_list = [current_m1]

    # --- 处理无人机方向向量 ---
    fy_dir_raw = np.array([fy_dir_x, fy_dir_y, 0.0])
    norm = np.linalg.norm(fy_dir_raw)
    
    if norm == 0:
        log_message = f"Invalid,{fy_speed_1},{fy_dir_x},{fy_dir_y},{drop_time_1},{clock_value_1},{drop_time_2},{clock_value_2},{drop_time_3},{clock_value_3},N/A (Zero Dir Vector)\n"
        log_file.write(log_message)
        log_file.flush()
        print(f"  Warning: Direction vector is zero for params: {params}. Returning large penalty.")
        return 1e10
    else:
        fy_dir_1 = fy_dir_raw / norm

    # --- 检查投弹时间是否在有效范围内 (边界约束已经处理，这里是额外的保护) ---
    # 注意：这里只检查了单个投弹时间，更严格的检查应该在约束函数中
    if not (0.1 <= drop_time_1 <= SIMULATION_DURATION - 0.1 and \
            0.1 <= drop_time_2 <= SIMULATION_DURATION - 0.1 and \
            0.1 <= drop_time_3 <= SIMULATION_DURATION - 0.1):
        log_message = f"Invalid,{fy_speed_1},{fy_dir_x},{fy_dir_y},{drop_time_1},{clock_value_1},{drop_time_2},{clock_value_2},{drop_time_3},{clock_value_3},N/A (Invalid Drop Time Range)\n"
        log_file.write(log_message)
        log_file.flush()
        print(f"  Warning: One or more drop_times out of valid range. Returning large penalty.")
        return 1e10

    # --- 创建无人机实例 ---
    fy_initial_pos = np.array([17800, 0, 1800])
    current_fy1 = Items.Drone(fy_initial_pos, fy_dir_1, fy_speed_1, 1)
    current_drone_list = [current_fy1]

    # --- 构建投掷事件字典 (包含 3 次投弹) ---
    # 为了确保 drop_events 字典的键是唯一的，并且事件能被正确触发，
    # 最好将 drop_time 稍微四舍五入到与 TIME_STEP 兼容的精度，或者使用更灵活的匹配逻辑。
    # utils.run_simulation 内部已经有 round(current_time_pre_calc, 5) 的逻辑，
    # 所以这里我们直接使用浮点数，并确保 drop_events 字典的键是唯一的。
    # 如果 drop_time_1, drop_time_2, drop_time_3 完全相等，可能会有问题。
    # 约束条件应该确保它们是不同的。
    drop_events = {
        drop_time_1: [(1, clock_value_1)],
        drop_time_2: [(1, clock_value_2)],
        drop_time_3: [(1, clock_value_3)]
    }
    # 确保 drop_events 字典的键是唯一的，如果优化器尝试生成重复的时间点，这会覆盖
    # 更好的做法是在约束中强制所有 drop_time 互不相同，或者在构建字典时处理冲突
    # 但由于我们有间隔约束，它们应该自然不同。

    # --- 打印当前参数 ---
    print(f"\n--- Running simulation with parameters ---")
    print(f"  fy_speed_1: {fy_speed_1:.4f}")
    print(f"  fy_dir_1 (raw): [{fy_dir_x:.4f}, {fy_dir_y:.4f}] -> (norm): {fy_dir_1}")
    print(f"  drop_time_1: {drop_time_1:.4f}, clock_value_1: {clock_value_1:.4f}")
    print(f"  drop_time_2: {drop_time_2:.4f}, clock_value_2: {clock_value_2:.4f}")
    print(f"  drop_time_3: {drop_time_3:.4f}, clock_value_3: {clock_value_3:.4f}")

    # --- 运行仿真 ---
    results = utils.run_simulation(
        current_missile_list,
        current_drone_list,
        static_target_samples,
        drop_events,
        TIME_STEP,
        SIMULATION_DURATION,
        SMOKE_RADIUS
    )

    # --- 获取优化目标值 ---
    obscured_duration = results.get('all_missiles_obscured', 0.0)
    print(f"  Obscured Duration: {obscured_duration:.4f}s")
    
    # 记录当前迭代的参数和结果到日志文件
    log_message = f"Iteration,{fy_speed_1},{fy_dir_x},{fy_dir_y},{drop_time_1},{clock_value_1},{drop_time_2},{clock_value_2},{drop_time_3},{clock_value_3},{obscured_duration}\n"
    log_file.write(log_message)
    log_file.flush() # 确保每次写入后都立即刷新到磁盘

    return -obscured_duration # 返回负值以进行最小化

# --- 定义约束函数 ---
# SLSQP 约束函数返回一个列表，每个元素代表一个约束
# 对于 'ineq' 约束，要求 func(x) >= 0
def drop_time_constraints(params):
    # 解包参数，只关心投弹时间
    _, _, _, drop_time_1, _, drop_time_2, _, drop_time_3, _ = params
    
    constraints = []
    # 约束 1: drop_time_2 - drop_time_1 >= 1.0
    constraints.append(drop_time_2 - drop_time_1 - 1.0)
    # 约束 2: drop_time_3 - drop_time_2 >= 1.0
    constraints.append(drop_time_3 - drop_time_2 - 1.0)
    
    return np.array(constraints)

# --- 定义优化起点 (Initial Guess) ---

# 初始猜测值需要包含所有 9 个参数
# 确保初始猜测满足约束条件 (drop_time_2 - drop_time_1 >= 1, drop_time_3 - drop_time_2 >= 1)
initial_guess = np.array([
    120.0,              # fy_speed_1
    -0.99993885,   # fy_dir_x
    0.01105904,   # fy_dir_y
    0.9758,                # drop_time_1
    3.8205,                # clock_value_1
    0.9758 + 1,          # drop_time_2 (确保大于 drop_time_1 + 1.0)
    5,                # clock_value_2
    0.9758 + 1 + 1,    # drop_time_3 (确保大于 drop_time_2 + 1.0)
    5                # clock_value_3
])

# --- 定义参数边界 ---
# 边界需要包含所有 9 个参数
bounds = [
    (70, 140),  # fy_speed_1
    (-1.0, 1.0),    # fy_dir_x
    (-1.0, 1.0),    # fy_dir_y
    (0.1, SIMULATION_DURATION - 0.1), # drop_time_1
    (0.1, 10.0),    # clock_value_1
    (0.1, SIMULATION_DURATION - 0.1), # drop_time_2
    (0.1, 10.0),    # clock_value_2
    (0.1, SIMULATION_DURATION - 0.1), # drop_time_3
    (0.1, 10.0)     # clock_value_3
]

# --- 定义约束对象 ---
# type: 'eq' (等于0) 或 'ineq' (大于等于0)
# fun: 约束函数
constraints = [{'type': 'ineq', 'fun': drop_time_constraints}]

# --- 运行优化 ---
print("\n--- Starting Optimization (SLSQP with Constraints) ---")
result = minimize(
    objective_for_minimize,
    initial_guess,
    method='SLSQP', # SLSQP 支持约束 L-BFGS-B
    bounds=bounds,
    constraints=constraints, # 传递约束
    options={
        'disp': True,       # 显示优化过程的详细信息
        'maxiter': 400,     # 最大迭代次数
        'ftol': 1e-6,       # 函数值收敛容差
        'eps': 1e-2         # 有限差分步长
    }
)

# --- 优化结束后关闭日志文件 ---
if log_file:
    log_file.close()
    print(f"\nOptimization log saved to: {log_filename}")
    log_file = None # 重置为 None

# --- 打印优化结果 ---
print("\n--- Optimization Results ---")

# 获取最终评估的参数和函数值 (无论成功与否)
final_evaluated_params = result.x
final_evaluated_obscured_duration = -result.fun

if result.success:
    optimized_params = result.x
    best_obscured_duration = -result.fun

    print(f"Optimization successful!")
    print(f"Best obscured duration found: {best_obscured_duration:.4f}s")
    print(f"Best parameters:")
    print(f"  fy_speed_1: {optimized_params[0]:.4f}")
    
    best_fy_dir_raw = np.array([optimized_params[1], optimized_params[2], 0.0])
    norm = np.linalg.norm(best_fy_dir_raw)
    best_fy_dir_1_normalized = best_fy_dir_raw / norm if norm != 0 else np.array([0.0, 0.0, 0.0])
    print(f"  fy_dir_1 (normalized): {best_fy_dir_1_normalized}")
    print(f"  drop_time_1: {optimized_params[3]:.4f}, clock_value_1: {optimized_params[4]:.4f}")
    print(f"  drop_time_2: {optimized_params[5]:.4f}, clock_value_2: {optimized_params[6]:.4f}")
    print(f"  drop_time_3: {optimized_params[7]:.4f}, clock_value_3: {optimized_params[8]:.4f}")
    print(f"Total function evaluations: {result.nfev}")
    print(f"Number of iterations: {result.nit}")
else:
    print(f"Optimization failed: {result.message}")
    print(f"Last evaluated parameters: {final_evaluated_params}")
    print(f"Last evaluated obscured duration: {final_evaluated_obscured_duration:.4f}s")
    print(f"Total function evaluations: {result.nfev}")
    print(f"Number of iterations: {result.nit}")
    
    optimized_params = final_evaluated_params # 即使失败也赋值，用于后续验证



