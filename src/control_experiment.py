"""
控制问题实验 - 阶段1：最小干预
=============================
目标：证明外部扰动可以强制系统从一个吸引子跳到另一个

控制项：u(t) * (target - M)
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/control', exist_ok=True)

# ==================== 1. 动力学（带控制）====================
def F_with_control(M_flat, t, omega, a_list, b_list, c_list, lambda_, mu, L, k, 
                   u_func, target_M, control_start, control_duration):
    M = M_flat.reshape((L, k))
    dM = np.zeros((L, k))
    
    # 自交互
    for l in range(L):
        for i in range(k):
            self_term = omega[i] * (a_list[i] * M[l,i]**3 + b_list[i] * M[l,i]**2 + c_list[i] * M[l,i])
            cross_group = -lambda_ * M[l,i] * np.sum(M[l, np.arange(k) != i])
            cross_layer = mu * np.sum(M[np.arange(L) != l, i])
            dM[l,i] = self_term + cross_group + cross_layer
    
    # 控制项
    if control_start <= t <= control_start + control_duration:
        u = u_func(t)
        current_M = M.flatten()
        control = u * (target_M - current_M)
        dM += control.reshape((L, k))
    
    return dM.flatten()

def run_with_control(Kc_list, lambda_, mu, u0, target_M, L=2, k=3, N_init=20):
    """运行控制实验，返回成功率"""
    omega = np.ones(k)
    a_list = [-3.0 / kc for kc in Kc_list]
    b_list = [4.5 / kc for kc in Kc_list]
    c_list = [-1.5 / kc for kc in Kc_list]
    
    success_count = 0
    T_total = 200
    control_start = 80
    control_duration = 50
    
    for _ in range(N_init):
        # 初始条件：随机
        init = np.random.uniform(0.01, 0.99, L * k)
        
        # 第一阶段：自由演化到稳态
        t1 = np.linspace(0, control_start, 500)
        sol1 = odeint(F_with_control, init, t1, 
                     args=(omega, a_list, b_list, c_list, lambda_, mu, L, k, 
                           lambda t: 0.0, target_M, control_start, control_duration))
        state_before = sol1[-1].copy()
        
        # 第二阶段：施加控制
        t2 = np.linspace(control_start, control_start + control_duration, 300)
        sol2 = odeint(F_with_control, state_before, t2,
                     args=(omega, a_list, b_list, c_list, lambda_, mu, L, k,
                           lambda t: u0, target_M, control_start, control_duration))
        state_during = sol2[-1].copy()
        
        # 第三阶段：自由演化到稳态
        t3 = np.linspace(control_start + control_duration, T_total, 500)
        sol3 = odeint(F_with_control, state_during, t3,
                     args=(omega, a_list, b_list, c_list, lambda_, mu, L, k,
                           lambda t: 0.0, target_M, control_start, control_duration))
        final_state = sol3[-1]
        
        # 检查是否到达目标吸引子（距离 target_M < 0.3）
        dist = np.sqrt(np.mean((final_state - target_M)**2))
        if dist < 0.3:
            success_count += 1
    
    return success_count / N_init

# ==================== 2. 主实验 ====================
print("=" * 60)
print("控制问题实验 - 阶段1：最小干预")
print("=" * 60)

# 参数
Kc_list = [0.32, 0.40, 0.48]  # 非对称
L, k = 2, 3
mu = 0.0

# 测试不同 λ
lambda_values = [0.1, 0.3, 0.5]
u0_values = np.linspace(0.0, 2.0, 21)

# 目标：切换到"极端态" M≈1.0
target_extreme = np.ones(L * k)

results = {}
for lam in lambda_values:
    results[lam] = []
    print(f"\nλ = {lam}")
    for u0 in u0_values:
        success = run_with_control(Kc_list, lam, mu, u0, target_extreme)
        results[lam].append((u0, success))
        if u0 in [0.0, 0.5, 1.0, 1.5, 2.0]:
            print(f"  u0 = {u0:.1f}: success = {success:.2f}")

# ==================== 3. 可视化 ====================
plt.figure(figsize=(10, 6))
for lam in lambda_values:
    us = [r[0] for r in results[lam]]
    ss = [r[1] for r in results[lam]]
    plt.plot(us, ss, 'o-', label=f'λ = {lam}', linewidth=2, markersize=5)

plt.xlabel('Control Strength u0', fontsize=12)
plt.ylabel('Success Rate', fontsize=12)
plt.title('Control: Success Rate vs Control Strength', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.05, 1.05)
plt.savefig('F:/hypergraph_bistability/figures/control/u0_vs_success.png', dpi=150)
plt.close()

# 找到阈值
print("\n" + "=" * 60)
print("最小控制阈值（成功率>50%）")
print("=" * 60)
for lam in lambda_values:
    for u0, success in results[lam]:
        if success > 0.5:
            print(f"λ = {lam}: u_crit ≈ {u0:.2f}")
            break

print("\n图片已保存到 figures/control/u0_vs_success.png")
