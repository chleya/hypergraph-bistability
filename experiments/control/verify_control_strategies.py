"""
控制策略系统验证
================

重新验证论文 Section 3.7 中的控制策略:
1. Global Pull: 全局加噪声/偏置
2. Global Boost: 全局增强目标
3. Local Boost: 局部精确增强
4. Random Local: 随机节点干预
5. Targeted Local (core/periphery): 针对边界

基于设计的多稳态模型:
dM_{i,l}/dt = α·M·(1-M)·(M-a) + λ·Σ(M_j - M_i) + μ·Σ(M_i,l' - M_i,l)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
import json

os.makedirs('F:/hypergraph_bistability/figures/control_verified', exist_ok=True)
os.makedirs('F:/hypergraph_bistability/results/control_verified', exist_ok=True)

# 参数
alpha = 1.0
a = 0.5  # 不稳定点
k, L = 3, 2  # 6维系统
Kc_list = [1.0] * k


def bistable_dynamics(t, M_flat, alpha, a, Kc_list, lambda_coupling, mu_coupling):
    """双稳态动力学"""
    M = M_flat.reshape(k, L)
    dMdt = np.zeros_like(M)
    
    for i in range(k):
        for l in range(L):
            Mc = M[i, l]
            # Bistable term
            bistable = alpha * Mc * (1 - Mc) * (Mc - a)
            
            # Group coupling
            group_coupling = 0
            for j in range(k):
                if j != i:
                    Mj_mean = np.mean(M[j, :])
                    group_coupling += lambda_coupling * (Mj_mean - Mc)
            
            # Layer coupling
            layer_coupling = 0
            for l2 in range(L):
                if l2 != l:
                    layer_coupling += mu_coupling * (M[i, l2] - Mc)
            
            dMdt[i, l] = bistable + group_coupling + layer_coupling
    
    return dMdt.flatten()


def apply_control_dynamics(t, M_flat, alpha, a, Kc_list, lambda_coupling, mu_coupling, 
                          control_type, control_params):
    """
    带控制的动力学
    control_type: 'global_pull', 'global_boost', 'local_boost', 'random_local', 'targeted_local'
    """
    M = M_flat.reshape(k, L)
    dMdt = np.zeros_like(M)
    
    for i in range(k):
        for l in range(L):
            Mc = M[i, l]
            bistable = alpha * Mc * (1 - Mc) * (Mc - a)
            
            group_coupling = 0
            for j in range(k):
                if j != i:
                    Mj_mean = np.mean(M[j, :])
                    group_coupling += lambda_coupling * (Mj_mean - Mc)
            
            layer_coupling = 0
            for l2 in range(L):
                if l2 != l:
                    layer_coupling += mu_coupling * (M[i, l2] - Mc)
            
            base_drift = bistable + group_coupling + layer_coupling
            
            # 应用控制
            control = 0
            if control_type == 'global_pull':
                # Global Pull: 添加指向目标状态的偏置
                target = control_params.get('target', 0.5)
                strength = control_params.get('strength', 0.1)
                control = strength * (target - Mc)
            
            elif control_type == 'global_boost':
                # Global Boost: 增强目标状态附近的吸引域
                # 使用更宽的 basin 吸引，不只是靠近目标时
                target = control_params.get('target', 1.0)
                strength = control_params.get('strength', 0.2)
                # 使用 sign 增益：当接近目标时增益更大，但远离时仍有效果
                distance = target - Mc
                basin_factor = 1.0 + 0.5 * np.exp(-((Mc - target)**2) / 0.3)
                control = strength * basin_factor * distance
            
            elif control_type == 'local_boost':
                # Local Boost: 只对特定 group 增强
                target_group = control_params.get('target_group', 0)
                target_layer = control_params.get('target_layer', 0)
                strength = control_params.get('strength', 0.3)
                if i == target_group and l == target_layer:
                    target = control_params.get('target', 1.0)
                    control = strength * (target - Mc)
            
            elif control_type == 'random_local':
                # Random Local: 预先选择要干预的节点，在整个控制阶段保持一致
                strength = control_params.get('strength', 0.2)
                if 'selected_mask' not in control_params:
                    n_nodes = k * L
                    control_params['selected_mask'] = np.random.random(n_nodes) < control_params.get('prob', 0.3)
                
                node_idx = i * L + l
                if control_params['selected_mask'][node_idx]:
                    target = control_params.get('target', 1.0)
                    control = strength * (target - Mc)
            
            elif control_type == 'targeted_local':
                # Targeted Local: 针对最有潜力的节点（最接近目标的节点）
                # 不只是边界，而是选择性地强化目标节点
                boundary_factor = control_params.get('boundary_factor', 1.0)
                strength = control_params.get('strength', 0.25)
                target = control_params.get('target', 1.0)
                
                # 选择当前最接近目标的节点进行强化
                M_flat_arr = M.flatten()
                distances = np.abs(M_flat_arr - target)
                n_target = max(1, int(control_params.get('target_ratio', 0.3) * k * L))
                target_indices = np.argsort(distances)[:n_target]
                node_idx = i * L + l
                if node_idx in target_indices:
                    control = strength * boundary_factor * (target - Mc)
            
            dMdt[i, l] = base_drift + control
    
    return dMdt.flatten()


def run_control_experiment(control_type, control_params, lambda_coupling, n_runs=50):
    """
    运行控制实验，返回成功率
    成功标准：最终状态收敛到目标吸引子
    """
    target_state = control_params.get('target', 1.0)  # 目标状态
    
    success_count = 0
    
    for run in range(n_runs):
        # 清除上次的随机选择状态
        if control_type == 'random_local':
            control_params.pop('selected_mask', None)
        elif control_type == 'targeted_local':
            control_params.pop('selected_indices', None)
        
        # 初始条件：从相反的 basin 开始
        if target_state > 0.5:
            # 目标是 HIGH，从 LOW 开始
            M0 = np.random.uniform(0.05, 0.25, k * L)
        else:
            # 目标是 LOW，从 HIGH 开始
            M0 = np.random.uniform(0.75, 0.95, k * L)
        
        # 第一阶段：自由演化到稳定 basin
        t1_end = 30
        sol1 = solve_ivp(
            lambda t, M: bistable_dynamics(t, M, alpha, a, Kc_list, lambda_coupling, 0.0),
            [0, t1_end], M0, method='RK45', rtol=1e-8, atol=1e-10
        )
        state_before = sol1.y[:, -1]
        
        # 第二阶段：施加控制
        t2_start = t1_end
        t2_end = t2_start + 30
        sol2 = solve_ivp(
            lambda t, M: apply_control_dynamics(t, M, alpha, a, Kc_list, lambda_coupling, 0.0,
                                               control_type, control_params),
            [t2_start, t2_end], state_before, method='RK45', rtol=1e-8, atol=1e-10
        )
        
        # 第三阶段：自由演化验证
        t3_end = t2_end + 50
        sol3 = solve_ivp(
            lambda t, M: bistable_dynamics(t, M, alpha, a, Kc_list, lambda_coupling, 0.0),
            [t2_end, t3_end], sol2.y[:, -1], method='RK45', rtol=1e-8, atol=1e-10
        )
        
        final_state = sol3.y[:, -1]
        M_final = final_state.reshape(k, L)
        
        # 不同策略有不同的成功标准
        # Global策略: 整个系统到达目标吸引子
        # Local策略: 目标节点/组显著高于其他节点
        if control_type in ['global_pull', 'global_boost']:
            final_mean = np.mean(final_state)
            if target_state > 0.5:
                if final_mean > 0.6:
                    success_count += 1
            else:
                if final_mean < 0.4:
                    success_count += 1
        elif control_type == 'local_boost':
            # 目标节点显著高于其他节点
            target_M = M_final[control_params['target_layer'], control_params['target_group']]
            others = [M_final[l, g] for l in range(L) for g in range(k) 
                     if not (l == control_params['target_layer'] and g == control_params['target_group'])]
            if target_M > np.mean(others) + 0.15:
                success_count += 1
        elif control_type in ['random_local', 'targeted_local']:
            # 这些策略目标不明确，按全局标准
            final_mean = np.mean(final_state)
            if target_state > 0.5:
                if final_mean > 0.6:
                    success_count += 1
            else:
                if final_mean < 0.4:
                    success_count += 1
    
    return success_count / n_runs


def verify_control_strategies():
    """验证所有控制策略"""
    print("=" * 60)
    print("Control Strategy Verification")
    print("=" * 60)
    
    lambda_values = [0.1, 0.3, 0.5]
    
    strategies = {
        'Global Pull': {
            'control_type': 'global_pull',
            'params': {'target': 1.0, 'strength': 0.15}
        },
        'Global Boost': {
            'control_type': 'global_boost', 
            'params': {'target': 1.0, 'strength': 0.2}
        },
        'Local Boost': {
            'control_type': 'local_boost',
            'params': {'target_group': 0, 'target_layer': 0, 'target': 1.0, 'strength': 0.3}
        },
        'Random Local': {
            'control_type': 'random_local',
            'params': {'target': 1.0, 'strength': 0.2, 'prob': 0.3}
        },
        'Targeted Local': {
            'control_type': 'targeted_local',
            'params': {'target': 1.0, 'strength': 0.25, 'boundary_factor': 1.5, 'target_ratio': 0.3}
        }
    }
    
    results = {}
    
    for strategy_name, config in strategies.items():
        print(f"\n{strategy_name}:")
        results[strategy_name] = {}
        
        for lam in lambda_values:
            success = run_control_experiment(
                config['control_type'],
                config['params'],
                lambda_coupling=lam,
                n_runs=50
            )
            results[strategy_name][f'λ={lam}'] = success
            print(f"  λ={lam}: success = {success:.1%}")
    
    return results


def plot_results(results):
    """绘制结果对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lambda_values = ['λ=0.1', 'λ=0.3', 'λ=0.5']
    x = np.arange(len(lambda_values))
    width = 0.15
    
    for i, (strategy_name, strategy_results) in enumerate(results.items()):
        successes = [strategy_results.get(lam, 0) for lam in lambda_values]
        ax.bar(x + i * width, successes, width, label=strategy_name)
    
    ax.set_xlabel('Coupling Strength (λ)')
    ax.set_ylabel('Success Rate')
    ax.set_title('Control Strategy Comparison')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(lambda_values)
    ax.legend(loc='upper right')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50% baseline')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('F:/hypergraph_bistability/figures/control_verified/strategy_comparison.png', dpi=150)
    print(f"\n[Saved] figures/control_verified/strategy_comparison.png")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Starting Control Strategy Verification")
    print("=" * 60)
    
    results = verify_control_strategies()
    plot_results(results)
    
    # 保存结果
    with open('F:/hypergraph_bistability/results/control_verified/verification_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for strategy, strategy_results in results.items():
        avg = np.mean(list(strategy_results.values()))
        print(f"{strategy}: avg success = {avg:.1%}")
