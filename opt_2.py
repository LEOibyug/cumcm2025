import numpy as np
import math
import Items
import utils
from tqdm import tqdm
from scipy.optimize import minimize
import datetime # 导入 datetime 模块

# --- 固定仿真参数 ---
TIME_STEP = 0.005
SIMULATION_DURATION = 15
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
def objective_for_minimize(params):
    global log_file # 声明使用全局变量

    if log_file is None: # 第一次调用时打开日志文件
        log_file = open(log_filename, 'w')
        log_file.write("Iteration,fy_speed_1,fy_dir_x,fy_dir_y,drop_time,clock_value,Obscured_Duration\n") # 写入CSV头部

    fy_speed_1, fy_dir_x, fy_dir_y, drop_time, clock_value = params

    current_m1 = Items.Missile(np.array([20000, 0, 2000]), fake_target, 1)
    current_missile_list = [current_m1]

    fy_dir_raw = np.array([fy_dir_x, fy_dir_y, 0.0])
    norm = np.linalg.norm(fy_dir_raw)
    
    if norm == 0:
        print(f"  Warning: Direction vector is zero for params: {params}. Returning large penalty.")
        # 记录到日志文件
        log_file.write(f"Invalid,{fy_speed_1},{fy_dir_x},{fy_dir_y},{drop_time},{clock_value},N/A (Zero Dir Vector)\n")
        log_file.flush() # 确保写入文件
        return 1e10
    else:
        fy_dir_1 = fy_dir_raw / norm

    if not (0.1 <= drop_time <= SIMULATION_DURATION - 0.1):
        print(f"  Warning: drop_time {drop_time:.2f} is out of valid range. Returning large penalty.")
        # 记录到日志文件
        log_file.write(f"Invalid,{fy_speed_1},{fy_dir_x},{fy_dir_y},{drop_time},{clock_value},N/A (Invalid Drop Time)\n")
        log_file.flush()
        return 1e10

    fy_initial_pos = np.array([17800, 0, 1800])
    current_fy1 = Items.Drone(fy_initial_pos, fy_dir_1, fy_speed_1, 1)
    current_drone_list = [current_fy1]

    drop_events = {drop_time: [(1, clock_value)]}

    print(f"\n--- Running simulation with parameters ---")
    print(f"  fy_speed_1: {fy_speed_1:.4f}")
    print(f"  fy_dir_1 (raw): [{fy_dir_x:.4f}, {fy_dir_y:.4f}] -> (norm): {fy_dir_1}")
    print(f"  drop_time: {drop_time:.4f}")
    print(f"  clock_value: {clock_value:.4f}")

    results = utils.run_simulation(
        current_missile_list,
        current_drone_list,
        static_target_samples,
        drop_events,
        TIME_STEP,
        SIMULATION_DURATION,
        SMOKE_RADIUS
    )

    obscured_duration = results.get('all_missiles_obscured', 0.0)
    print(f"  Obscured Duration: {obscured_duration:.4f}s")
    
    # 记录当前迭代的参数和结果到日志文件
    log_file.write(f"Iteration,{fy_speed_1},{fy_dir_x},{fy_dir_y},{drop_time},{clock_value},{obscured_duration}\n")
    log_file.flush() # 确保每次写入后都立即刷新到磁盘

    return -obscured_duration

# --- 定义优化起点 (Initial Guess) --

initial_guess = np.array([
    119.7736,              # fy_speed_1
    -0.99993885,   # fy_dir_x
    0.01105904,   # fy_dir_y
    0.9758,                # drop_time_1
    3.8205,                # clock_value_1
])

# --- 定义参数边界 ---
bounds = [
    (70, 140),
    (-1.0, 1.0),
    (0, 1.0),
    (0.1, SIMULATION_DURATION - 0.1),
    (0.1, 10.0)
]

# --- 运行优化 ---
print("\n--- Starting Optimization (L-BFGS-B) ---")
result = minimize(
    objective_for_minimize,
    initial_guess,
    method='L-BFGS-B', # 使用 L-BFGS-B 或 SLSQP
    bounds=bounds,
    options={
        'disp': True,
        'maxiter': 400,
        'ftol': 1e-6,
        'eps': 1e-2
    }
)

# --- 优化结束后关闭日志文件 ---
if log_file:
    log_file.close()
    print(f"\nOptimization log saved to: {log_filename}")

# --- 打印优化结果 ---
print("\n--- Optimization Results ---")

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
    print(f"  drop_time: {optimized_params[3]:.4f}")
    print(f"  clock_value: {optimized_params[4]:.4f}")
    print(f"Total function evaluations: {result.nfev}")
else:
    print(f"Optimization failed: {result.message}")
    print(f"Last evaluated parameters: {final_evaluated_params}")
    print(f"Last evaluated obscured duration: {final_evaluated_obscured_duration:.4f}s")
    print(f"Total function evaluations: {result.nfev}")
    
    optimized_params = final_evaluated_params



# 再次关闭日志文件，以防验证调用又打开了它
if log_file:
    log_file.close()
    log_file = None # 重置为 None，避免下次运行时出错
