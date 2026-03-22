"""
Path B: Microscopic Hypergraph Model Verification
=================================================

验证问题：微观超图规则（growth/split/fusion）是否涌现出与理论模型相同的
多稳态结构？即：
1. 张量积结构 N_att = 2^{k×L} at λ=0
2. λ 引起的吸引子坍缩
3. μ 的同步/竞争效应

方法：
1. 用理论模型（bistable drift）找到不同 basin 的代表性初始条件
2. 用这些初始条件在微观超图模型中验证
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.integrate import solve_ivp

os.makedirs('F:/hypergraph_bistability/results/verification_b', exist_ok=True)
os.makedirs('F:/hypergraph_bistability/figures/verification_b', exist_ok=True)

from multi_stability.multi_stability_core import (
    MultiGroupHypergraph,
    MultiLayerMultiGroupHypergraph
)


# ============================================================================
# Part 1: Theoretical Model - Find Representative Initial Conditions
# ============================================================================

alpha = 1.0
a = 0.5

def theoretical_dynamics(M_flat, k, L, lambda_coupling, mu_coupling):
    """理论双稳态动力学"""
    M = np.array(M_flat).reshape(k, L)
    dMdt = np.zeros((k, L))
    for i in range(k):
        for l in range(L):
            Mc = float(M[i, l])
            bistable = alpha * Mc * (1.0 - Mc) * (Mc - a)
            group_coupling = 0
            for j in range(k):
                if j != i:
                    Mj = float(np.mean(M[j, :]))
                    group_coupling += lambda_coupling * (Mj - Mc)
            layer_coupling = 0
            for l2 in range(L):
                if l2 != l:
                    layer_coupling += mu_coupling * (M[i, l2] - Mc)
            dMdt[i, l] = bistable + group_coupling + layer_coupling
    return dMdt.flatten()


def find_basin_representatives(k, L, lambda_coupling=0.0, mu_coupling=0.0, n_samples=500):
    """
    用理论模型找每个 basin 的代表性初始条件
    返回: dict {binary_state: representative_M0}
    """
    basin_centers = {}
    
    dyn = lambda t, M: theoretical_dynamics(M, k, L, lambda_coupling, mu_coupling)
    
    for _ in range(n_samples):
        M0 = np.random.uniform(0.05, 0.95, k * L)
        try:
            sol = solve_ivp(dyn, [0, 100], M0, method='RK45', rtol=1e-10, atol=1e-12)
            final = sol.y[:, -1]
            # Classify as binary
            binary = tuple(0 if v < 0.5 else 1 for v in final)
            if binary not in basin_centers:
                basin_centers[binary] = M0
        except:
            pass
    
    return basin_centers


# ============================================================================
# Part 2: Microscopic Hypergraph Model with Controlled Initialization
# ============================================================================

def create_hypergraph_with_activity(k, L, N=50, gamma=0.35, 
                                    inter_group_competition=0.0,
                                    inter_layer_coupling=0.0,
                                    init_activity=None, seed=42):
    """
    创建超图，并控制初始活跃度
    
    init_activity: dict {(i,l): activity} 
                   activity ∈ [0, 1], 控制每个 group-layer 的初始超边数量
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if init_activity is None:
        init_activity = {(i, l): 0.5 for i in range(k) for l in range(L)}
    
    # 创建多层超图
    system = MultiLayerMultiGroupHypergraph(
        N=N,
        n_groups=k,
        n_layers=L,
        gamma=gamma,
        state_dim=16,
        inter_group_competition=inter_group_competition,
        inter_layer_coupling=inter_layer_coupling,
        seed=seed
    )
    
    # 调整初始超边数量来控制活跃度
    for layer_idx in range(L):
        layer = system.layers[layer_idx]
        for group_idx in range(k):
            target_activity = init_activity.get((group_idx, layer_idx), 0.5)
            group_nodes = layer.get_group_nodes(group_idx)
            
            if not group_nodes:
                continue
            
            # 调整超边数量
            # 目标：用超边数量来调制度分布
            n_edges_to_add = int(target_activity * 10) - len([e for e in layer.E 
                                                             if all(layer.get_node_group(v) == group_idx for v in e)])
            
            for _ in range(abs(n_edges_to_add)):
                if n_edges_to_add > 0:
                    # 添加超边
                    if len(group_nodes) >= 2:
                        size = random.randint(2, min(5, len(group_nodes) // 2))
                        e = frozenset(random.sample(group_nodes, size))
                        if e not in layer.E:
                            layer.E.append(e)
                else:
                    # 删除超边
                    group_edges = [e for e in layer.E 
                                  if all(layer.get_node_group(v) == group_idx for v in e)]
                    if group_edges:
                        layer.E.remove(random.choice(group_edges))
    
    return system


def get_hypergraph_state(system):
    """获取超图的序参量状态"""
    states = []
    for l in range(system.n_layers):
        for g in range(system.n_groups):
            M = system.layers[l].get_group_order_parameter(g)
            states.append(M)
    return np.array(states)


def run_hypergraph_experiment(k, L, lambda_coupling, mu_coupling,
                              n_runs_per_basin=10, N=50, T=120):
    """
    运行超图实验：用每个 basin 的代表性初始条件
    """
    # 找理论模型的 basin 代表性
    basin_reps = find_basin_representatives(k, L, lambda_coupling=0.0, mu_coupling=0.0)
    
    results = {}
    all_final_states = []
    all_initial_activities = []
    
    for basin, M0_representative in basin_reps.items():
        basin_results = []
        
        for run in range(n_runs_per_basin):
            seed = hash((basin, run)) % 1000000
            
            # 从代表性 M0 创建初始活跃度
            # M0 ∈ [0,1] -> 映射到超边数量因子
            activity = {}
            for idx, (i, l) in enumerate([(i, l) for i in range(k) for l in range(L)]):
                activity[(i, l)] = M0_representative[idx]
            
            # 创建超图
            hg = create_hypergraph_with_activity(
                k=k, L=L, N=N, gamma=0.35,
                inter_group_competition=lambda_coupling,
                inter_layer_coupling=mu_coupling,
                init_activity=activity,
                seed=seed
            )
            
            # 运行动力学
            history = hg.run_dynamics(steps=T)
            
            # 获取最终状态
            final_state = get_hypergraph_state(hg)
            all_final_states.append(final_state)
            all_initial_activities.append(activity)
            
            # 判断收敛到哪个 basin
            final_binary = tuple(0 if v < 0.3 else (1 if v > 0.7 else 2) for v in final_state)
            basin_results.append({
                'initial_activity': activity,
                'final_state': final_state,
                'final_binary': final_binary
            })
        
        results[basin] = basin_results
    
    return results, all_final_states, all_initial_activities


# ============================================================================
# Part 3: Simplified Direct Test
# ============================================================================

def direct_hypergraph_test(k, L, lambda_coupling, mu_coupling, 
                           n_runs=100, N=50, T=100):
    """
    直接测试：随机初始化超图，统计吸引子数
    """
    final_states = []
    
    for run in range(n_runs):
        seed = 42 + run
        
        hg = MultiLayerMultiGroupHypergraph(
            N=N,
            n_groups=k,
            n_layers=L,
            gamma=0.35,
            state_dim=16,
            inter_group_competition=lambda_coupling,
            inter_layer_coupling=mu_coupling,
            seed=seed
        )
        
        history = hg.run_dynamics(steps=T)
        final_state = get_hypergraph_state(hg)
        final_states.append(final_state)
    
    return np.array(final_states)


def cluster_states_vector(states, threshold=0.15):
    """对 k×L 维状态向量聚类"""
    unique = []
    for state in states:
        is_new = True
        for u in unique:
            dist = np.sqrt(np.mean((state - u)**2))
            if dist < threshold:
                is_new = False
                break
        if is_new:
            unique.append(state)
    return len(unique), unique


# ============================================================================
# Main Verification
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Path B: Microscopic Hypergraph Model Verification")
    print("=" * 60)
    
    # Step 1: Direct test at λ=0, μ=0
    print("\nStep 1: Direct Test at λ=0, μ=0")
    print("-" * 40)
    
    configs = [
        (1, 1),
        (2, 1),
        (1, 2),
        (2, 2),
    ]
    
    direct_results = {}
    for k, L in configs:
        print(f"\nTesting k={k}, L={L}...")
        states = direct_hypergraph_test(k, L, lambda_coupling=0.0, mu_coupling=0.0, n_runs=50)
        n_att, unique = cluster_states_vector(states, threshold=0.15)
        predicted = 2 ** (k * L)
        match = 'PASS' if n_att == predicted else 'FAIL'
        direct_results[(k, L)] = {
            'n_attractors': n_att,
            'predicted': predicted,
            'match': match
        }
        print(f"  N_attractors: {n_att}, Predicted: {predicted}, [{match}]")
    
    # Step 2: Test λ effect
    print("\n\nStep 2: λ Effect (k=2, L=1)")
    print("-" * 40)
    
    lambda_results = {}
    for lam in [0.0, 0.2, 0.5, 1.0]:
        states = direct_hypergraph_test(2, 1, lambda_coupling=lam, mu_coupling=0.0, n_runs=50)
        n_att, unique = cluster_states_vector(states, threshold=0.15)
        lambda_results[lam] = n_att
        print(f"  λ={lam:.1f}: N_attractors = {n_att}")
    
    # Step 3: Test μ effect
    print("\n\nStep 3: μ Effect (k=2, L=2, λ=0)")
    print("-" * 40)
    
    mu_results = {}
    for mu in [-0.2, 0.0, 0.2]:
        states = direct_hypergraph_test(2, 2, lambda_coupling=0.0, mu_coupling=mu, n_runs=50)
        n_att, unique = cluster_states_vector(states, threshold=0.15)
        mu_results[mu] = n_att
        print(f"  μ={mu:.1f}: N_attractors = {n_att}")
    
    # Save results
    verification_results = {
        'direct_test': {f'k{k}_L{L}': v for (k, L), v in direct_results.items()},
        'lambda_effect': {str(k): v for k, v in lambda_results.items()},
        'mu_effect': {str(k): v for k, v in mu_results.items()}
    }
    
    with open('F:/hypergraph_bistability/results/verification_b/results.json', 'w') as f:
        json.dump(verification_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n1. Tensor Product Test (λ=0, μ=0):")
    for (k, L), v in direct_results.items():
        print(f"   k={k}, L={L}: {v['n_att']} attractors (expected {v['predicted']}) {v['match']}")
    
    print(f"\n2. λ Effect (k=2, L=1):")
    for lam, n_att in lambda_results.items():
        print(f"   λ={lam:.1f}: {n_att} attractors")
    
    print(f"\n3. μ Effect (k=2, L=2, λ=0):")
    for mu, n_att in mu_results.items():
        print(f"   μ={mu:.1f}: {n_att} attractors")
