"""
PPO Local Control 验证
=======================
问题：PPO的62%成功是在非标准dynamics下测量的
目标：在标准bistable dynamics下验证local控制是否有效

标准设置：
- dynamics: alpha * M * (1-M) * (M-a)
- Kc_list: [0.35, 0.35, 0.35] (对称)
- boost机制: additive push (和Global Pull/Boost对比)

非标准设置(PPO原始):
- dynamics: a*M^3 + b*M^2 + c*M
- Kc_list: [0.32, 0.40, 0.48] (非对称)
- boost机制: 参数修改
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/results/PPO_verification', exist_ok=True)

ALPHA = 1.0
A = 0.5
K = 3
L = 2
T_SETTLE = 50.0


def ode_standard(t, M_flat, k, L, lam, mu):
    """标准 bistable dynamics + coupling"""
    M = M_flat.reshape(k, L)
    dM = np.zeros_like(M)
    for i in range(k):
        for l in range(L):
            Mc = M[i, l]
            bistable = ALPHA * Mc * (1-Mc) * (Mc - A)
            gc = sum(lam * (np.mean(M[j,:]) - Mc) for j in range(k) if j != i)
            lc = sum(mu * (M[i,l2] - Mc) for l2 in range(L) if l2 != l)
            dM[i, l] = bistable + gc + lc
    return dM.flatten()


def ode_nonstandard(M_flat, t, a_list, b_list, c_list, lambda_, mu, L, k, boost_info):
    """PPO使用的非标准dynamics"""
    M = M_flat.reshape((L, k))
    dM = np.zeros((L, k))
    
    target_layer, target_group, boost_factor, active = boost_info
    
    a_mod = a_list.copy()
    b_mod = b_list.copy()
    c_mod = c_list.copy()
    
    if active:
        a_mod[target_group] /= boost_factor
        b_mod[target_group] /= boost_factor
        c_mod[target_group] /= boost_factor
    
    for l in range(L):
        for i in range(k):
            self_term = a_mod[i] * M[l,i]**3 + b_mod[i] * M[l,i]**2 + c_mod[i] * M[l,i]
            cross_group = -lambda_ * M[l,i] * np.sum(M[l, np.arange(k) != i])
            cross_layer = mu * np.sum(M[np.arange(L) != l, i])
            dM[l,i] = self_term + cross_group + cross_layer
    
    return dM.flatten()


def run_standard_experiment(lam, target, boost_start, boost_duration, boost_strength, 
                           n_init=30, mu=0.0):
    """标准设置下的local boost实验"""
    success_count = 0
    
    for seed in range(n_init):
        np.random.seed(seed)
        
        # 初始状态：LOW态 (全部~0.05)
        state = np.full((K, L), 0.05)
        state[target] = 0.95  # target初始化为HIGH
        
        # 展平
        state_flat = state.flatten()
        
        # 阶段1：自由演化到boost_start
        if boost_start > 0:
            sol1 = solve_ivp(
                lambda t, M: ode_standard(t, M, K, L, lam, mu),
                [0, boost_start], state_flat,
                method='RK45', rtol=1e-4, atol=1e-6
            )
            state_flat = sol1.y[:, -1]
        
        # 阶段2：boost (additive push to target)
        if boost_duration > 0 and boost_strength > 0:
            boost_times = np.linspace(0, boost_duration, 20)
            for dt in boost_times:
                dM = ode_standard(0, state_flat, K, L, lam, mu)
                state_flat = state_flat + boost_strength * dM * dt
                state_flat = np.clip(state_flat, 0, 1)
        
        # 阶段3：自由演化
        sol3 = solve_ivp(
            lambda t, M: ode_standard(t, M, K, L, lam, mu),
            [0, T_SETTLE], state_flat,
            method='RK45', rtol=1e-4, atol=1e-6
        )
        final_state = sol3.y[:, -1].reshape(K, L)
        
        # 检查成功：target是否为HIGH
        target_M = final_state[target]
        other_mean = np.mean([final_state[i,j] for i in range(K) for j in range(L) 
                             if (i,j) != target])
        
        if target_M > 0.7 and target_M > other_mean + 0.1:
            success_count += 1
    
    return success_count / n_init


def run_nonstandard_experiment(lam, Kc_list, target, boost_start, boost_duration, boost_strength,
                               n_init=30, mu=0.0):
    """非标准设置(PPO原始)下的实验"""
    L_k = 2
    k = 3
    
    a_list = [-3.0 / kc for kc in Kc_list]
    b_list = [4.5 / kc for kc in Kc_list]
    c_list = [-1.5 / kc for kc in Kc_list]
    
    success_count = 0
    target_layer, target_group = target
    
    for seed in range(n_init):
        np.random.seed(seed)
        
        # 随机初始状态
        state = np.random.uniform(0.1, 0.9, L_k * k)
        
        t_max = 200.0
        
        # 阶段1
        if boost_start > 0:
            boost_info_1 = (target_layer, target_group, 1.0, False)
            t1 = np.linspace(0, boost_start, 50)
            from scipy.integrate import odeint
            sol1 = odeint(ode_nonstandard, state, t1,
                         args=(a_list, b_list, c_list, lam, mu, L_k, k, boost_info_1))
            state = sol1[-1].copy()
        
        # 阶段2
        if boost_duration > 0 and boost_strength > 1.0:
            boost_info_2 = (target_layer, target_group, boost_strength, True)
            t2 = np.linspace(0, boost_duration, 50)
            sol2 = odeint(ode_nonstandard, state, t2,
                         args=(a_list, b_list, c_list, lam, mu, L_k, k, boost_info_2))
            state = sol2[-1].copy()
        
        # 阶段3
        boost_info_3 = (target_layer, target_group, 1.0, False)
        t3 = np.linspace(0, t_max, 100)
        sol3 = odeint(ode_nonstandard, state, t3,
                     args=(a_list, b_list, c_list, lam, mu, L_k, k, boost_info_3))
        
        final_M = sol3[-1].reshape((L_k, k))
        target_M = final_M[target_layer, target_group]
        other_values = [final_M[l, g] for l in range(L_k) for g in range(k) 
                       if (l, g) != (target_layer, target_group)]
        avg_other = np.mean(other_values)
        
        if target_M > avg_other + 0.15:
            success_count += 1
    
    return success_count / n_init


def main():
    print("=" * 60)
    print("PPO Local Control 验证")
    print("=" * 60)
    
    results = {
        'standard': {},
        'nonstandard': {}
    }
    
    lambda_values = [0.05, 0.1, 0.2]
    target = (0, 0)
    
    # PPO学到的策略
    ppo_boost_start = 40  # 中间时刻
    ppo_boost_duration = 30
    ppo_boost_strength = 1.5
    
    # 固定local boost (additive)
    fixed_boost_start = 10
    fixed_boost_duration = 50
    fixed_boost_strength = 0.3  # 较小持续推力
    
    print("\n[1] 标准dynamics + 固定additive boost")
    print("-" * 50)
    for lam in lambda_values:
        success = run_standard_experiment(
            lam, target, 
            fixed_boost_start, fixed_boost_duration, fixed_boost_strength,
            n_init=20
        )
        results['standard']['fixed'] = results['standard'].get('fixed', {})
        results['standard']['fixed'][lam] = success
        print(f"  λ={lam}: success = {success:.0%}")
    
    print("\n[2] 标准dynamics + PPO学习boost (待训练)")
    print("-" * 50)
    print("  需要完整PPO训练，skipping...")
    
    print("\n[3] 非标准dynamics (PPO原始)")
    print("-" * 50)
    Kc_list = [0.32, 0.40, 0.48]
    for lam in lambda_values:
        success = run_nonstandard_experiment(
            lam, Kc_list, target,
            ppo_boost_start, ppo_boost_duration, ppo_boost_strength,
            n_init=20
        )
        results['nonstandard'][lam] = success
        print(f"  λ={lam}: success = {success:.0%}")
    
    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    print(f"{'λ':>6s} | {'标准fixed':>12s} | {'非标准PPO':>12s}")
    print("-" * 40)
    for lam in lambda_values:
        std_fixed = results['standard']['fixed'].get(lam, 0)
        nonstd = results['nonstandard'].get(lam, 0)
        print(f"{lam:>6.2f} | {std_fixed:>11.0%} | {nonstd:>11.0%}")
    
    print("""
结论:
- 62%来自非标准dynamics + 非对称Kc
- 标准dynamics下local控制效果需要单独验证
- 需要训练PPO在标准设置下看是否能学到有效策略
    """)


if __name__ == '__main__':
    main()