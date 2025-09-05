import numpy as np
import math
import Items
import utils
from tqdm import tqdm

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import datetime
import os

# --- 固定仿真参数 ---
TIME_STEP = 0.005
SIMULATION_DURATION = 40 # 仿真总时长，需要足够长以容纳3个投弹事件
SMOKE_RADIUS = 10

# --- 静态和固定参与者 ---
fake_target = Items.Plot(np.array([0, 0, 0]))
target = Items.Volume(np.array([0, 200, 5]), 7, 10)

static_target_samples = []
for samples in target.get_sample():
    static_target_samples.append(samples)

# --- 设置日志文件 ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/optimization_log_bayesian_multi_drop_{timestamp}.csv"
os.makedirs("logs", exist_ok=True) # 确保 logs 文件夹存在

# --- 定义优化目标函数 (使用 @use_named_args 装饰器) ---
# 增加了 drop_time_2_interval, drop_time_3_interval, clock_value_2, clock_value_3
@use_named_args([
    Real(70.0, 140.0, name='fy_speed_1'),
    Real(-1.0, 1.0, name='fy_dir_x'),
    Real(0.0, 1.0, name='fy_dir_y'),
    Real(0.1, SIMULATION_DURATION - 2.1, name='drop_time_1'), # drop_time_1 必须留出足够空间给后续事件
    Real(0.1, 10.0, name='clock_value_1'),
    Real(1.0, 5.0, name='drop_time_2_interval'), # 第二个事件与第一个事件的间隔 (>=1s)
    Real(0.1, 10.0, name='clock_value_2'),
    Real(1.0, 5.0, name='drop_time_3_interval'), # 第三个事件与第二个事件的间隔 (>=1s)
    Real(0.1, 10.0, name='clock_value_3')
])
def objective_for_bayesian(fy_speed_1, fy_dir_x, fy_dir_y,
                           drop_time_1, clock_value_1,
                           drop_time_2_interval, clock_value_2,
                           drop_time_3_interval, clock_value_3):

    # 计算实际的投弹时间
    actual_drop_time_2 = drop_time_1 + drop_time_2_interval
    actual_drop_time_3 = actual_drop_time_2 + drop_time_3_interval

    # 检查实际投弹时间是否在仿真范围内
    # 尽管 space 已经做了初步限制，这里做最终检查
    if not (0.1 <= actual_drop_time_2 <= SIMULATION_DURATION - 0.1 and
            0.1 <= actual_drop_time_3 <= SIMULATION_DURATION - 0.1):
        print(f"  Warning: Actual drop times out of valid range. Returning large penalty.")
        with open(log_filename, 'a') as log_file:
            if log_file.tell() == 0: log_file.write("Iteration,fy_speed_1,fy_dir_x,fy_dir_y,dt1,cv1,dt2_int,cv2,dt3_int,cv3,Actual_dt2,Actual_dt3,Obscured_Duration\n")
            log_file.write(f"Invalid,{fy_speed_1},{fy_dir_x},{fy_dir_y},{drop_time_1},{clock_value_1},{drop_time_2_interval},{clock_value_2},{drop_time_3_interval},{clock_value_3},{actual_drop_time_2},{actual_drop_time_3},N/A (Invalid Actual Drop Time)\n")
            log_file.flush()
        return 1e10

    with open(log_filename, 'a') as log_file:
        if log_file.tell() == 0:
            log_file.write("Iteration,fy_speed_1,fy_dir_x,fy_dir_y,dt1,cv1,dt2_int,cv2,dt3_int,cv3,Actual_dt2,Actual_dt3,Obscured_Duration\n")

        current_m1 = Items.Missile(np.array([20000, 0, 2000]), fake_target, 1)
        current_missile_list = [current_m1]

        fy_dir_raw = np.array([fy_dir_x, fy_dir_y, 0.0])
        norm = np.linalg.norm(fy_dir_raw)
        
        if norm == 0:
            print(f"  Warning: Direction vector is zero. Returning large penalty.")
            log_file.write(f"Invalid,{fy_speed_1},{fy_dir_x},{fy_dir_y},{drop_time_1},{clock_value_1},{drop_time_2_interval},{clock_value_2},{drop_time_3_interval},{clock_value_3},{actual_drop_time_2},{actual_drop_time_3},N/A (Zero Dir Vector)\n")
            log_file.flush()
            return 1e10
        else:
            fy_dir_1 = fy_dir_raw / norm

        fy_initial_pos = np.array([17800, 0, 1800])
        current_fy1 = Items.Drone(fy_initial_pos, fy_dir_1, fy_speed_1, 1)
        current_drone_list = [current_fy1]

        # 构建投掷事件字典，包含所有三个事件
        drop_events = {
            drop_time_1: [(1, clock_value_1)],
            actual_drop_time_2: [(1, clock_value_2)],
            actual_drop_time_3: [(1, clock_value_3)]
        }
        # 确保事件按时间顺序处理，虽然 utils.run_simulation 内部会排序，但这里明确一下
        # 也可以将 drop_events 的键四舍五入到某个精度，以避免浮点数比较问题
        # 例如: drop_events = {round(t, 4): v for t, v in drop_events.items()}


        print(f"\n--- Running simulation with parameters ---")
        print(f"  fy_speed_1: {fy_speed_1:.4f}")
        print(f"  fy_dir_1 (raw): [{fy_dir_x:.4f}, {fy_dir_y:.4f}] -> (norm): {fy_dir_1}")
        print(f"  drop_time_1: {drop_time_1:.4f}")
        print(f"  clock_value_1: {clock_value_1:.4f}")
        print(f"  drop_time_2_interval: {drop_time_2_interval:.4f} -> Actual drop_time_2: {actual_drop_time_2:.4f}")
        print(f"  clock_value_2: {clock_value_2:.4f}")
        print(f"  drop_time_3_interval: {drop_time_3_interval:.4f} -> Actual drop_time_3: {actual_drop_time_3:.4f}")
        print(f"  clock_value_3: {clock_value_3:.4f}")


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
        
        log_file.write(f"Evaluation,{fy_speed_1},{fy_dir_x},{fy_dir_y},{drop_time_1},{clock_value_1},{drop_time_2_interval},{clock_value_2},{drop_time_3_interval},{clock_value_3},{actual_drop_time_2},{actual_drop_time_3},{obscured_duration}\n")
        log_file.flush()

        return -obscured_duration

# --- 定义参数空间 ---
# 增加了 drop_time_2_interval, drop_time_3_interval, clock_value_2, clock_value_3
space = [
    Real(70.0, 140.0, name='fy_speed_1'),
    Real(-1.0, 1.0, name='fy_dir_x'),
    Real(0.0, 1.0, name='fy_dir_y'),
    Real(0.1, SIMULATION_DURATION - 2.1, name='drop_time_1'), # 确保 dt1 后面至少有 2 个 1s 间隔
    Real(0.1, 10.0, name='clock_value_1'),
    Real(1.0, 5.0, name='drop_time_2_interval'), # 间隔至少1s，最大5s
    Real(0.1, 10.0, name='clock_value_2'),
    Real(1.0, 5.0, name='drop_time_3_interval'), # 间隔至少1s，最大5s
    Real(0.1, 10.0, name='clock_value_3')
]

# --- 定义初始点 (可选，用于引导贝叶斯优化) ---
# 需要提供一个包含 9 个参数的初始点
# 示例：基于你之前的单投弹最佳点进行扩展
# fy_speed_1: 103.6163
# fy_dir_x: 0.9956142
# fy_dir_y: 0.0935541
# drop_time_1: 0.8725
# clock_value_1: 0.2591
# 假设后续事件间隔1s，clock_value 相同
initial_points = [
    [120, -1, 0, # 速度和方向
     1.5, 3.6,                 # drop_time_1, clock_value_1
     1.0, 5,                    # drop_time_2_interval (1s), clock_value_2
     1.0, 5]                    # drop_time_3_interval (1s), clock_value_3
]

# 确保 SIMULATION_DURATION 足够长，以容纳所有投弹事件
# 如果 drop_time_1 = 0.1, interval_1 = 5, interval_2 = 5, 那么最后一个投弹时间是 0.1 + 5 + 5 = 10.1s
# 所以 SIMULATION_DURATION = 15s 是足够的

# --- 运行贝叶斯优化 ---
print("\n--- Starting Bayesian Optimization ---")
res_gp = gp_minimize(
    objective_for_bayesian,
    space,
    x0=initial_points,
    n_calls=1000,             # 增加总评估次数，因为参数维度增加了
    n_random_starts=10,      # 增加随机探索次数
    random_state=42,
    acq_func="EI",
    acq_optimizer="auto",
    n_jobs=-1,
    verbose=True,
)

# --- 打印优化结果 ---
print("\n--- Optimization Results ---")

optimized_params = res_gp.x
best_obscured_duration = -res_gp.fun

print(f"Optimization successful!")
print(f"Best obscured duration found: {best_obscured_duration:.4f}s")
print(f"Best parameters:")
print(f"  fy_speed_1: {optimized_params[0]:.4f}")

best_fy_dir_raw = np.array([optimized_params[1], optimized_params[2], 0.0])
norm = np.linalg.norm(best_fy_dir_raw)
best_fy_dir_1_normalized = best_fy_dir_raw / norm if norm != 0 else np.array([0.0, 0.0, 0.0])
print(f"  fy_dir_1 (normalized): {best_fy_dir_1_normalized}")

# 打印实际的投弹时间
actual_drop_time_2_best = optimized_params[3] + optimized_params[5]
actual_drop_time_3_best = actual_drop_time_2_best + optimized_params[7]

print(f"  drop_time_1: {optimized_params[3]:.4f}")
print(f"  clock_value_1: {optimized_params[4]:.4f}")
print(f"  drop_time_2_interval: {optimized_params[5]:.4f} -> Actual drop_time_2: {actual_drop_time_2_best:.4f}")
print(f"  clock_value_2: {optimized_params[6]:.4f}")
print(f"  drop_time_3_interval: {optimized_params[7]:.4f} -> Actual drop_time_3: {actual_drop_time_3_best:.4f}")
print(f"  clock_value_3: {optimized_params[8]:.4f}")

print(f"Total function evaluations: {res_gp.n_calls_}")

# 可选：再次运行最佳参数的仿真，以验证结果
print("\n--- Verifying best parameters ---")
# 传入参数时需要解包 optimized_params
final_obscured_duration_check = objective_for_bayesian(*optimized_params)
print(f"Verification result (should match best found): {-final_obscured_duration_check:.4f}s")

