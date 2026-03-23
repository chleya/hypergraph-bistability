"""
控制问题实验 - 方向2：局部 Precision Boost
==========================================
目标：只对目标群体/层施加 boost，而不是全局

原理：更精准的控制，减少干扰
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/control', exist_ok=True)

# ==================== 1. 动力学（局部 precision boost）====================
def F_with_local_boost(M_flat, t, omega, a_list_base, b_list_base, c_list_base, 
                       lambda_, mu, L, k, boost_params):
    """
    boost_params = (target_layer, target_group, boost_factor, T_start, T_end)
    只对目标层-目标群体施加 boost
    """
    M = M_flat.reshape((L, k))
    dM = np.zeros((L, k))
    
    # 动态修改（只对目标层-目标群体）
    target_layer, target_group, boost_factor, T_start, T_end = boost_params
    
    # 复制基础参数
    a_list = a_list_base.copy()
    b_list = b_list_base.copy()
    c_list = c_list_base.copy()
    
    # 只对目标施加 boost
    if T_start <= t <= T_end:
        a_list[target_group] /= boost_factor
        b_list[target_group] /= boost_factor
        c_list[target_group] /= boost_factor
    
    # 动力学
    for l in range(L):
        for i in range(k):
            self_term = omega[i] * (a_list[i] * M[l,i]**3 + b_list[i] * M[l,i]**2 + c_list[i] * M[l,i])
            cross_group = -lambda_ * M[l,i] * np.sum(M[l, np.arange(k) != i])
            cross_layer = mu * np.sum(M[np.arange(L) != l, i])
            dM[l,i] = self_term + cross_group + cross_layer
    
    return dM.flatten()

def run_local_boost_experiment(Kc_list, lambda_, boost_factor, target_layer, target_group, L=2, k=3, N_init=40):
    """运行局部 boost 实验"""
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
        # 初始条件：偏向其他群体
        init = np.random.uniform(0.1, 0.9, L * k)
        
        # 第一阶段：自由演化
        t1 = np.linspace(0, boost_start, 300)
        sol1 = odeint(F_with_local_boost, init, t1,
                     args=(omega, a_list_base, b_list_base, c_list_base, lambda_, mu, L, k,
                           (target_layer, target_group, boost_factor, boost_start, boost_end)))
        state_before = sol1[-1].copy()
        
        # 第二阶段：局部 boost
        t2 = np.linspace(boost_start, boost_end, 200)
        sol2 = odeint(F_with_local_boost, state_before, t2,
                     args=(omega, a_list_base, b_list_base, c_list_base, lambda_, mu, L, k,
                           (target_layer, target_group, boost_factor, boost_start, boost_end)))
        
        # 第三阶段：自由演化
        t3 = np.linspace(boost_end, T_total, 500)
        sol3 = odeint(F_with_local_boost, sol2[-1], t3,
                     args=(omega, a_list_base, b_list_base, c_list_base, lambda_, mu, L, k,
                           (target_layer, target_group, boost_factor, boost_start, boost_end)))
        final_state = sol3[-1]
        
        # 检查目标层-群体是否主导
        M = final_state.reshape((L, k))
        target_M = M[target_layer, target_group]
        # 与其他所有群体比较
        other_values = [M[l, g] for l in range(L) for g in range(k) if (l, g) != (target_layer, target_group)]
        avg_other = np.mean(other_values)
        
        if target_M > avg_other + 0.15:  # 目标显著高于其他
            success_count += 1
    
    return success_count / N_init

# ==================== 2. 主实验 ====================
print("=" * 60)
print("控制实验 - 方向2：局部 Precision Boost")
print("=" * 60)

# 参数
Kc_list = [0.32, 0.40, 0.48]  # 非对称
L, k = 2, 3
mu = 0.0

# 测试不同 λ
lambda_values = [0.1, 0.3, 0.5]
boost_factors = [1.0, 1.5, 2.0, 2.5, 3.0]

results = {}
for lam in lambda_values:
    results[lam] = []
    print(f"\nλ = {lam}")
    for bf in boost_factors:
        best_success = 0
        best_combo = None
        for target_layer in range(L):
            for target_group in range(k):
                success = run_local_boost_experiment(Kc_list, lam, bf, target_layer, target_group)
                if success > best_success:
                    best_success = success
                    best_combo = (target_layer, target_group)
        results[lam].append((bf, best_success, best_combo))
        if bf in [1.0, 1.5, 2.0, 2.5, 3.0]:
            print(f"  boost = {bf:.1f}: success = {best_success:.2f} (layer={best_combo[0]}, group={best_combo[1]})")

# ==================== 3. 可视化 ====================
plt.figure(figsize=(10, 6))
for lam in lambda_values:
    bfs = [r[0] for r in results[lam]]
    ss = [r[1] for r in results[lam]]
    plt.plot(bfs, ss, 'o-', label=f'λ = {lam}', linewidth=2, markersize=8)

plt.xlabel('Boost Factor (Local)', fontsize=12)
plt.ylabel('Success Rate', fontsize=12)
plt.title('Control: Local Precision Boost (Only Target Group)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.05, 1.05)
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
plt.savefig('F:/hypergraph_bistability/figures/control/local_boost_vs_success.png', dpi=150)
plt.close()

# 找到阈值
print("\n" + "=" * 60)
print("最小 Boost Factor（成功率>50%）")
print("=" * 60)
for lam in lambda_values:
    found = False
    for bf, success, _ in results[lam]:
        if success > 0.5:
            print(f"λ = {lam}: boost_crit ≈ {bf:.1f} (success = {success:.2f})")
            found = True
            break
    if not found:
        max_s = max([r[1] for r in results[lam]])
        print(f"λ = {lam}: no threshold (max = {max_s:.2f})")

print("\n图片已保存到 figures/control/local_boost_vs_success.png")
