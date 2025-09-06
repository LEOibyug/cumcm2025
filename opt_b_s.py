import numpy as np
import math
import Items
import utils
from tqdm import tqdm
# from scipy.optimize import minimize # 不再需要 scipy.optimize.minimize

# 导入 scikit-optimize 相关的模块
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver # 用于保存优化过程
import datetime # 导入 datetime 模块
import os # 用于文件路径操作

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
log_filename = f"logs/optimization_log_bayesian_{timestamp}.csv" # 修改日志文件名为 CSV 格式
# 日志文件将在 objective_for_bayesian 内部打开和写入，每次写入后刷新

# --- 定义优化目标函数 (使用 @use_named_args 装饰器) ---
@use_named_args([
    Real(1.873, 10, name='drop_t'),
    Real(0, 10, name='clock'),
])
def objective_for_bayesian(drop_t, clock):
    specified_params = [103.6163, 0.9956142, 0.0935541,0.8725,0.2591]
    # 每次调用时打开并追加写入日志文件，然后关闭
    # 这样可以避免全局变量的复杂性，并确保每次评估都被记录
    # 但如果函数评估非常频繁，可能会有性能开销，可以考虑在回调中集中写入
    # 这里为了简单，我们每次都打开关闭
    with open(log_filename, 'a') as log_file:
        # 检查文件是否为空，如果是则写入头部
        if log_file.tell() == 0:
            log_file.write("Iteration,fy_speed_1,fy_dir_x,fy_dir_y,drop_time,clock_value,Obscured_Duration\n")

        # 重新创建导弹实例，确保每次仿真都从初始状态开始
        current_m1 = Items.Missile(np.array([20000, 0, 2000]), fake_target, 1)
        current_missile_list = [current_m1]

        # 构建无人机方向向量并归一化
        fy_dir_raw = np.array([specified_params[1], specified_params[2], 0.0])
        norm = np.linalg.norm(fy_dir_raw)
        
        if norm == 0:
            return 1e10
        else:
            fy_dir_1 = fy_dir_raw / norm


        # 重新创建无人机实例
        fy_initial_pos = np.array([17800, 0, 1800])
        current_fy1 = Items.Drone(fy_initial_pos, fy_dir_1, specified_params[0], 1)
        current_drone_list = [current_fy1]

        # 构建投掷事件字典
        drop_events = {specified_params[3]: [(1, specified_params[4])], 
                       drop_t: [(1, clock)]
                        }

        # --- 打印当前参数 ---
        print(f"\n--- Running simulation with parameters ---")
        print(f"  fy_speed_1: {specified_params[0]:.4f}")
        print(f"  fy_dir_1 (raw): [{specified_params[1]:.4f}, {specified_params[2]:.4f}] -> (norm): {fy_dir_1}")
        print(f"  drop_events: {drop_events}")

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

        obscured_duration = results.get('all_missiles_obscured', 0.0)
        print(f"  Obscured Duration: {obscured_duration:.4f}s")
        
        # 记录当前迭代的参数和结果到日志文件
        # 注意：这里没有迭代计数器，skopt 的回调函数可以提供
        log_file.write(f"Evaluation,{drop_t},{clock},{obscured_duration}\n")
        log_file.flush() # 确保每次写入后都立即刷新到磁盘

        return -obscured_duration # 返回负值以进行最小化

# --- 定义参数空间 (与 @use_named_args 装饰器中的定义一致) ---
space = [
    Real(1.873, 10, name='drop_t'),
    Real(0, 10, name='clock'),
]

# --- 定义初始点 (可选，用于引导贝叶斯优化) ---
# skopt 接受一个列表的列表作为初始点，每个内层列表是一个参数组合
# 我们可以使用你提供的 initial_guess 作为唯一的初始点
initial_points = [
    [1.874,0.0]
]
#    [119.7736, -0.99993885, 0.01105904, 0.9758, 3.8205]



# --- 运行贝叶斯优化 ---
print("\n--- Starting Bayesian Optimization ---")
res_gp = gp_minimize(
    objective_for_bayesian,
    space,
    x0=initial_points,      # 初始评估点
    n_calls=500,             # 总的函数评估次数 (包括 x0 中的点)
    n_random_starts=5,      # 初始随机探索的次数 (不包括 x0 中的点)
                            # n_calls = len(x0) + n_random_starts + n_iterations_after_random_starts
    random_state=42,        # 随机种子，用于结果复现
    acq_func="EI",          # 采集函数，"EI" (Expected Improvement) 是常用且效果不错的选择
    acq_optimizer="auto",   # 采集函数优化器
    n_jobs=-1,              # 使用所有可用CPU核心并行评估
    verbose=True,           # 显示优化过程信息
)

# --- 打印优化结果 ---
print("\n--- Optimization Results ---")

# res_gp.x 包含最佳参数组合
# res_gp.fun 包含最佳目标函数值 (最小化的负值)
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
print(f"  drop_time: {optimized_params[3]:.4f}")
print(f"  clock_value: {optimized_params[4]:.4f}")

# 可选：再次运行最佳参数的仿真，以验证结果
print("\n--- Verifying best parameters ---")
# 注意：这里直接调用 objective_for_bayesian 会再次打印参数和运行仿真
# 传入参数时需要解包 optimized_params
final_obscured_duration_check = objective_for_bayesian(*optimized_params)
print(f"Verification result (should match best found): {-final_obscured_duration_check:.4f}s")

