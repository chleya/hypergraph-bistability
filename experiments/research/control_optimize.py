"""
控制优化实验 - 更全面的参数扫描
================================
目标：找到最佳参数组合以提高成功率
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/control', exist_ok=True)

# ==================== 动力学 ====================
def F_with_boost(M_flat, t, omega, a_list_base, b_list_base, c_list_base, 
                lambda_, mu, L, k, boost_params):
    """带 boost 的动力学"""
    M = M_flat.reshape((L, k))
    dM = np.zeros((L, k))
    
    target_layer, target_group, boost_factor, T_start, T_end = boost_params
    
    a_list = a_list_base.copy()
    b_list = b_list_base.copy()
    c_list = c_list_base.copy()
    
    if T_start <= t <= T_end:
        a_list[target_group] /= boost_factor
        b_list[target_group] /= boost_factor
        c_list[target_group] /= boost_factor
    
    for l in range(L):
        for i in range(k):
            self_term = omega[i] * (a_list[i] * M[l,i]**3 + b_list[i] * M[l,i]**2 + c_list[i] * M[l,i])
            cross_group = -lambda_ * M[l,i] * np.sum(M[l, np.arange(k) != i])
            cross_layer = mu * np.sum(M[np.arange(L) != l, i])
            dM[l,i] = self_term + cross_group + cross_layer
    
    return dM.flatten()

def run_experiment(Kc_list, lambda_, boost_factor, boost_start, boost_duration, 
                  target_layer, target_group, L=2, k=3, N_init=50):
    """运行单个实验"""
    omega = np.ones(k)
    a_list_base = [-3.0 / kc for kc in Kc_list]
    b_list_base = [4.5 / kc for kc in Kc_list]
    c_list_base = [-1.5 / kc for kc in Kc_list]
    
    T_total = 250
    boost_end = boost_start + boost_duration
    
    success_count = 0
    
    for _ in range(N_init):
        init = np.random.uniform(0.1, 0.9, L * k)
        
        t1 = np.linspace(0, boost_start, 300)
        sol1 = odeint(F_with_boost, init, t1,
                     args=(omega, a_list_base, b_list_base, c_list_base, lambda_, mu, L, k,
                           (target_layer, target_group, boost_factor, boost_start, boost_end)))
        
        t2 = np.linspace(boost_start, boost_end, 200)
        sol2 = odeint(F_with_boost, sol1[-1], t2,
                     args=(omega, a_list_base, b_list_base, c_list_base, lambda_, mu, L, k,
                           (target_layer, target_group, boost_factor, boost_start, boost_end)))
        
        t3 = np.linspace(boost_end, T_total, 500)
        sol3 = odeint(F_with_boost, sol2[-1], t3,
                     args=(omega, a_list_base, b_list_base, c_list_base, lambda_, mu, L, k,
                           (target_layer, target_group, boost_factor, boost_start, boost_end)))
        
        M = sol3[-1].reshape((L, k))
        target_M = M[target_layer, target_group]
        other_values = [M[l, g] for l in range(L) for g in range(k) if (l, g) != (target_layer, target_group)]
        avg_other = np.mean(other_values)
        
        if target_M > avg_other + 0.15:
            success_count += 1
    
    return success_count / N_init

# ==================== 主实验 ====================
print("=" * 60)
print("控制优化 - 全面参数扫描")
print("=" * 60)

Kc_list = [0.32, 0.40, 0.48]
L, k = 2, 3
mu = 0.0

# 固定在最佳 λ
lambda_fixed = 0.1

# 扫描参数
boost_factors = [1.3, 1.5, 1.7, 2.0]
boost_starts = [30, 50, 70]
boost_durations = [20, 30, 50]

results = []
best_success = 0
best_params = None

print(f"\nλ = {lambda_fixed}")
print("-" * 50)

for bf in boost_factors:
    for bs in boost_starts:
        for bd in boost_durations:
            # 测试所有目标组合
            best_for_combo = 0
            for tl in range(L):
                for tg in range(k):
                    success = run_experiment(Kc_list, lambda_fixed, bf, bs, bd, tl, tg)
                    best_for_combo = max(best_for_combo, success)
            
            results.append({
                'boost_factor': bf,
                'boost_start': bs,
                'boost_duration': bd,
                'success': best_for_combo
            })
            
            if best_for_combo > best_success:
                best_success = best_for_combo
                best_params = (bf, bs, bd)
            
            print(f"bf={bf:.1f}, start={bs}, dur={bd}: {best_for_combo:.2f}")

# ==================== 可视化 ====================
# 整理数据
df_data = {}
for r in results:
    bf = r['boost_factor']
    if bf not in df_data:
        df_data[bf] = {'starts': [], 'durs': [], 'success': []}
    df_data[bf]['starts'].append(r['boost_start'])
    df_data[bf]['durs'].append(r['boost_duration'])
    df_data[bf]['success'].append(r['success'])

# 找到最佳
print("\n" + "=" * 60)
print("最佳参数组合")
print("=" * 60)
print(f"boost_factor = {best_params[0]}")
print(f"boost_start = {best_params[1]}")
print(f"boost_duration = {best_params[2]}")
print(f"success_rate = {best_success:.2f}")

# 绘制热图
for bf in boost_factors:
    matrix = np.zeros((len(boost_starts), len(boost_durations)))
    for i, bs in enumerate(boost_starts):
        for j, bd in enumerate(boost_durations):
            for r in results:
                if r['boost_factor'] == bf and r['boost_start'] == bs and r['boost_duration'] == bd:
                    matrix[i, j] = r['success']
    
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(label='Success Rate')
    plt.xticks(range(len(boost_durations)), boost_durations)
    plt.yticks(range(len(boost_starts)), boost_starts)
    plt.xlabel('Boost Duration')
    plt.ylabel('Boost Start')
    plt.title(f'Control Optimization (λ={lambda_fixed}, boost_factor={bf})')
    plt.savefig(f'F:/hypergraph_bistability/figures/control/optimize_bf{bf}.png', dpi=150)
    plt.close()

print("\n图片已保存")
