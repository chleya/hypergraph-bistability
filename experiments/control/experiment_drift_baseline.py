"""
k=1, L=1 Baseline with Drift Function
====================================

验证当使用显式立方漂移函数 F(M) = α(M-M1*)(M-M0)(M-M2*) 时，
k=1, L=1 是否产生双稳态。

这才是真正的基线测试。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

os.makedirs('F:/hypergraph_bistability/figures/dimension_study', exist_ok=True)

# ============================================================================
# 漂移函数模型（来自 bridge_fm.py）
# ============================================================================

M1_star = 0.45
M0 = 0.60
M2_star = 1.0

def F(M, alpha=1.0, noise=0.0):
    """理论 drift 函数: F(M) = α(M-M1*)(M-M0)(M-M2*)"""
    drift = alpha * (M - M1_star) * (M - M0) * (M - M2_star)
    if noise > 0:
        drift += np.random.normal(0, noise)
    return drift


def simulate_drift(a=1.0, b=0.15, x0=None, steps=100, dt=0.1, noise=0.01):
    """
    模拟漂移动力学: dx/dt = F(x)
    对应 minimal_model.py 的 model_simulation
    """
    if x0 is None:
        x0 = np.random.uniform(0.1, 0.9)
    
    x = x0
    trajectory = [x]
    
    for _ in range(steps):
        # 用 Euler 方法
        # 但这里我们用 F(M) 的立方形式
        dM = F(x, alpha=a, noise=noise)
        x = x + dt * dM
        x = np.clip(x, 0.001, 0.999)  # 保持在水池内
        trajectory.append(x)
    
    return np.array(trajectory)


def count_attractors(trajectories, tolerance=0.1):
    """聚类轨迹终点来计数吸引子"""
    final_values = [t[-1] for t in trajectories]
    
    unique = []
    for v in final_values:
        is_new = True
        for u in unique:
            if abs(v - u) < tolerance:
                is_new = False
                break
        if is_new:
            unique.append(v)
    
    return len(unique), unique


# ============================================================================
# 实验1: k=1, L=1 基线（单群体单层）
# ============================================================================

def experiment_k1_L1():
    """k=1, L=1 基线测试"""
    print("=" * 60)
    print("Experiment 1: k=1, L=1 with Drift Function")
    print("=" * 60)
    
    n_runs = 50
    steps = 100
    dt = 0.1
    
    trajectories = []
    final_M = []
    
    for run in range(n_runs):
        x0 = np.random.uniform(0.1, 0.9)
        traj = simulate_drift(
            a=1.0, 
            x0=x0, 
            steps=steps, 
            dt=dt, 
            noise=0.0  # 先用无噪声
        )
        trajectories.append(traj)
        final_M.append(traj[-1])
        
        if (run + 1) % 10 == 0:
            print(f"  Run {run + 1}/{n_runs}")
    
    n_att, unique = count_attractors(trajectories, tolerance=0.1)
    
    print(f"\n[Results - No Noise]")
    print(f"  n_attractors: {n_att}")
    print(f"  unique M values: {sorted(unique)}")
    
    return trajectories, final_M, n_att, unique


# ============================================================================
# 实验2: 不同初始条件的更细粒度扫描
# ============================================================================

def experiment_fine_grained():
    """更细粒度的初始条件扫描"""
    print("\n" + "=" * 60)
    print("Experiment 2: Fine-grained Initial Condition Scan")
    print("=" * 60)
    
    x0_range = np.linspace(0.01, 0.99, 30)
    final_M = []
    
    for x0 in x0_range:
        traj = simulate_drift(a=1.0, x0=x0, steps=100, dt=0.1, noise=0.0)
        final_M.append(traj[-1])
    
    final_M = np.array(final_M)
    
    # 找两个簇
    n_att, unique = count_attractors([final_M], tolerance=0.1)
    
    print(f"  Initial conditions scanned: {len(x0_range)}")
    print(f"  n_attractors: {n_att}")
    print(f"  unique final M: {sorted(unique)}")
    
    return x0_range, final_M, n_att, unique


# ============================================================================
# 实验3: 加噪声后看 basin 跳跃
# ============================================================================

def experiment_noisy():
    """加噪声看 basin 跳跃"""
    print("\n" + "=" * 60)
    print("Experiment 3: Noisy Dynamics")
    print("=" * 60)
    
    n_runs = 50
    final_M_noisy = []
    
    for run in range(n_runs):
        x0 = np.random.uniform(0.1, 0.9)
        traj = simulate_drift(a=1.0, x0=x0, steps=100, dt=0.1, noise=0.02)
        final_M_noisy.append(traj[-1])
    
    final_M_noisy = np.array(final_M_noisy)
    n_att, unique = count_attractors([final_M_noisy], tolerance=0.1)
    
    print(f"  n_runs with noise=0.02: {n_runs}")
    print(f"  n_attractors: {n_att}")
    print(f"  unique final M: {sorted(unique)}")
    print(f"  Mean final M: {final_M_noisy.mean():.3f} ± {final_M_noisy.std():.3f}")
    
    return final_M_noisy, n_att, unique


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    # 实验1: 无噪声基线
    trajs1, final1, n_att1, unique1 = experiment_k1_L1()
    
    # 实验2: 细粒度扫描
    x0s, final2, n_att2, unique2 = experiment_fine_grained()
    
    # 实验3: 加噪声
    final3, n_att3, unique3 = experiment_noisy()
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: 无噪声轨迹
    ax1 = axes[0, 0]
    for traj in trajs1[:20]:
        ax1.plot(traj, alpha=0.5, linewidth=0.8)
    ax1.axhline(M1_star, color='blue', linestyle='--', alpha=0.7, label=f'M1*={M1_star}')
    ax1.axhline(M2_star, color='red', linestyle='--', alpha=0.7, label=f'M2*={M2_star}')
    ax1.axhline(M0, color='gray', linestyle=':', alpha=0.7, label=f'M0={M0}')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('M')
    ax1.set_title(f'k=1 L=1 No Noise (n_att={n_att1})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 细粒度扫描 - 初始 vs 最终
    ax2 = axes[0, 1]
    ax2.scatter(x0s, final2, alpha=0.6, s=50)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax2.axhline(M1_star, color='blue', linestyle='--', alpha=0.7)
    ax2.axhline(M2_star, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Initial M')
    ax2.set_ylabel('Final M')
    ax2.set_title(f'Initial vs Final (n_att={n_att2})')
    ax2.grid(True, alpha=0.3)
    
    # 图3: 最终M分布直方图
    ax3 = axes[1, 0]
    ax3.hist(final1, bins=20, alpha=0.7, edgecolor='black', label='No noise')
    ax3.hist(final3, bins=20, alpha=0.5, edgecolor='black', label='With noise')
    ax3.axvline(M1_star, color='blue', linestyle='--', linewidth=2)
    ax3.axvline(M2_star, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Final M')
    ax3.set_ylabel('Count')
    ax3.set_title('Final M Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 图4: F(M) 函数曲线
    ax4 = axes[1, 1]
    M_range = np.linspace(0, 1.1, 200)
    F_values = [F(m, alpha=1.0) for m in M_range]
    ax4.plot(M_range, F_values, 'b-', linewidth=2)
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.axvline(M1_star, color='blue', linestyle='--', alpha=0.7, label=f'M1*={M1_star}')
    ax4.axvline(M0, color='gray', linestyle=':', alpha=0.7, label=f'M0={M0}')
    ax4.axvline(M2_star, color='red', linestyle='--', alpha=0.7, label=f'M2*={M2_star}')
    ax4.set_xlabel('M')
    ax4.set_ylabel('F(M)')
    ax4.set_title('Drift Function F(M)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('k=1 L=1 Baseline with Cubic Drift Function', fontsize=14)
    plt.tight_layout()
    
    save_path = 'F:/hypergraph_bistability/figures/dimension_study/k1_L1_drift_baseline.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[Saved] {save_path}")
    plt.close()
    
    # 总结
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"k=1 L=1 (no noise): n_att = {n_att1}")
    print(f"k=1 L=1 (fine-grained): n_att = {n_att2}")
    print(f"k=1 L=1 (with noise): n_att = {n_att3}")
    print(f"\nExpected: 2 attractors (M1*≈0.45, M2*≈1.0)")
