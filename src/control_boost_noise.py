"""
控制问题实验 - 方向1：Precision Boost + 小噪声共振
====================================================
目标：在临界区加入小噪声帮助翻越势垒

原理：临界区势垒最低，噪声能帮助"翻过最后一厘米"
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/control', exist_ok=True)

# ==================== 1. 动力学（带 precision boost + 噪声）====================
def F_with_boost_and_noise(M_flat, t, omega, a_list_base, b_list_base, c_list_base, 
                           lambda_, mu, L, k, boost_params, sigma):
    """
    boost_params = (target_group, boost_factor, T_start, T_end)
    sigma = noise amplitude
    """
    M = M_flat.reshape((L, k))
    dM = np.zeros((L, k))
    
    # 动态修改 a,b,c（precision boost）
    target_group, boost_factor, T_start, T_end = boost_params
    if T_start <= t <= T_end:
        a_list = a_list_base.copy()
        b_list = b_list_base.copy()
        c_list = c_list_base.copy()
        a_list[target_group] /= boost_factor
        b_list[target_group] /= boost_factor
        c_list[target_group] /= boost_factor
    else:
        a_list = a_list_base
        b_list = b_list_base
        c_list = c_list_base
    
    # 动力学
    for l in range(L):
        for i in range(k):
            self_term = omega[i] * (a_list[i] * M[l,i]**3 + b_list[i] * M[l,i]**2 + c_list[i] * M[l,i])
            cross_group = -lambda_ * M[l,i] * np.sum(M[l, np.arange(k) != i])
            cross_layer = mu * np.sum(M[np.arange(L) != l, i])
            dM[l,i] = self_term + cross_group + cross_layer
    
    # 添加噪声（只在 boost 期间）
    if T_start <= t <= T_end and sigma > 0:
        noise = sigma * np.random.randn(L * k)
        dM_flat = dM.flatten()
        dM_flat += noise
        dM = dM_flat.reshape((L, k))
    
    return dM.flatten()

def run_boost_noise_experiment(Kc_list, lambda_, sigma, boost_factor, target_group, L=2, k=3, N_init=40):
    """运行 boost + noise 实验"""
    omega = np.ones(k)
    a_list_base = [-3.0 / kc for kc in Kc_list]
    b_list_base = [4.5 / kc for kc in Kc_list]
    c_list_base = [-1.5 / kc for kc in Kc_list]
    
    T_total = 200
    boost_start = 50
    boost_duration = 30
    boost_end = boost_start + boost_duration
    
    success_count = 0
    
    for _ in range(N_init):
        init = np.random.uniform(0.1, 0.9, L * k)
        
        # 第一阶段：自由演化
        t1 = np.linspace(0, boost_start, 300)
        sol1 = odeint(F_with_boost_and_noise, init, t1,
                     args=(omega, a_list_base, b_list_base, c_list_base, lambda_, mu, L, k,
                           (target_group, boost_factor, boost_start, boost_end), 0.0))
        state_before = sol1[-1].copy()
        
        # 第二阶段：boost + noise
        t2 = np.linspace(boost_start, boost_end, 200)
        sol2 = odeint(F_with_boost_and_noise, state_before, t2,
                     args=(omega, a_list_base, b_list_base, c_list_base, lambda_, mu, L, k,
                           (target_group, boost_factor, boost_start, boost_end), sigma))
        
        # 第三阶段：自由演化
        t3 = np.linspace(boost_end, T_total, 500)
        sol3 = odeint(F_with_boost_and_noise, sol2[-1], t3,
                     args=(omega, a_list_base, b_list_base, c_list_base, lambda_, mu, L, k,
                           (target_group, boost_factor, boost_start, boost_end), 0.0))
        final_state = sol3[-1]
        
        # 检查目标群体是否主导
        M = final_state.reshape((L, k))
        target_M = M[0, target_group]
        avg_other = np.mean([M[0, i] for i in range(k) if i != target_group])
        
        if target_M > avg_other + 0.1:
            success_count += 1
    
    return success_count / N_init

# ==================== 2. 主实验 ====================
print("=" * 60)
print("控制实验 - Precision Boost + 噪声共振")
print("=" * 60)

# 参数
Kc_list = [0.32, 0.40, 0.48]
L, k = 2, 3
mu = 0.0
lambda_fixed = 0.3  # 固定在临界区

# 扫描 boost 和 sigma
boost_factors = [1.8, 2.0, 2.2]
sigma_values = [0.0, 0.01, 0.02, 0.03]

results = {}
print(f"\nλ = {lambda_fixed} (临界区)")
print("-" * 40)

for sigma in sigma_values:
    results[sigma] = []
    for bf in boost_factors:
        best_success = 0
        for target_group in range(k):
            success = run_boost_noise_experiment(Kc_list, lambda_fixed, sigma, bf, target_group)
            best_success = max(best_success, success)
        results[sigma].append((bf, best_success))
        print(f"σ = {sigma:.2f}, boost = {bf:.1f}: success = {best_success:.2f}")

# ==================== 3. 可视化 ====================
plt.figure(figsize=(10, 6))
for sigma in sigma_values:
    bfs = [r[0] for r in results[sigma]]
    ss = [r[1] for r in results[sigma]]
    plt.plot(bfs, ss, 'o-', label=f'σ = {sigma:.2f}', linewidth=2, markersize=8)

plt.xlabel('Boost Factor', fontsize=12)
plt.ylabel('Success Rate', fontsize=12)
plt.title(f'Control: Precision Boost + Noise Resonance (λ={lambda_fixed})', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.05, 1.05)
plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='70% target')
plt.savefig('F:/hypergraph_bistability/figures/control/boost_noise_resonance.png', dpi=150)
plt.close()

# 找到最佳组合
print("\n" + "=" * 60)
print("最佳组合（成功率>70%）")
print("=" * 60)
best_found = False
for sigma in sigma_values:
    for bf, success in results[sigma]:
        if success > 0.7:
            print(f"✅ σ = {sigma:.2f}, boost = {bf:.1f}: success = {success:.2f}")
            best_found = True

if not best_found:
    # 找最高
    max_success = 0
    best_combo = None
    for sigma in sigma_values:
        for bf, success in results[sigma]:
            if success > max_success:
                max_success = success
                best_combo = (sigma, bf)
    print(f"最高成功率: σ = {best_combo[0]:.2f}, boost = {best_combo[1]:.1f}: {max_success:.2f}")

print("\n图片已保存")
