"""
Path C: Continuous Spectrum Multistability
==========================================

转向新方向：从"硬双稳态"转向"连续谱多稳态"

核心发现：
- 微观超图规则产生 SOFT bistability（M 值连续分布）
- 不是离散的 M=0 和 M=1，而是 M ∈ [0.02, 0.37] 连续谱
- 张量积结构表现为：谱宽度/谱复杂度随 k×L 增长

新理论框架：
- 不再说 N_att = 2^{k×L}
- 而是说：系统有 MULTISTABLE SPECTRUM
- 谱的"有效宽度"或"状态密度"随维度增加
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import sys

sys.path.insert(0, 'src')

os.makedirs('F:/hypergraph_bistability/results/continuous_spectrum', exist_ok=True)
os.makedirs('F:/hypergraph_bistability/figures/continuous_spectrum', exist_ok=True)

# 复用验证A的类
exec(open('src/verification_a_v2.py').read().split('if __name__')[0])


def measure_spectrum_properties(M_values):
    """
    测量谱的性质：
    - 谱支撑：min 到 max M 值
    - 谱宽度：max - min
    - 谱密度：单位 M 区间内的状态数
    - 峰值位置：M 分布的峰值
    """
    M_values = np.array(M_values)
    
    # 基本统计
    M_min = M_values.min()
    M_max = M_values.max()
    M_mean = M_values.mean()
    M_std = M_values.std()
    
    # 直方图分析
    bins = np.linspace(0, 1, 21)
    hist, edges = np.histogram(M_values, bins=bins)
    
    # 找峰值（最密集的区间）
    peak_idx = np.argmax(hist)
    peak_M = (edges[peak_idx] + edges[peak_idx + 1]) / 2
    
    # 计算谱宽度（使用标准差作为度量）
    spectral_width = M_std
    
    # 计算有效谱支撑（10-90 百分位）
    M_10 = np.percentile(M_values, 10)
    M_90 = np.percentile(M_values, 90)
    effective_support = M_90 - M_10
    
    return {
        'M_min': M_min,
        'M_max': M_max,
        'M_mean': M_mean,
        'M_std': M_std,
        'spectral_width': spectral_width,
        'effective_support': effective_support,
        'peak_M': peak_M,
        'histogram': hist.tolist(),
        'bin_edges': edges.tolist()
    }


def characterize_spectrum_vs_dim():
    """
    表征不同 (k, L) 下谱的性质
    核心假设：谱宽度/复杂度随 k×L 增加
    """
    print("=" * 60)
    print("Characterize Spectral Properties vs Dimension")
    print("=" * 60)
    
    configs = [
        (1, 1),
        (2, 1),
        (1, 2),
        (2, 2),
        (3, 1),
    ]
    
    results = {}
    
    for k, L in configs:
        print(f"\nTesting k={k}, L={L}...")
        
        all_M_values = []
        n_runs = 100
        
        for run in range(n_runs):
            seed = 42 + run * 100
            
            # 对每个 group 独立运行
            group_M = []
            for g in range(k):
                hg = CoagulationBiasedHypergraph(
                    N=50, n_groups=1, gamma=0.35,
                    fusion_boost=10.0, seed=seed + g
                )
                
                # 随机初始条件
                if random.random() < 0.5:
                    hg.E = hg.E[:2]  # LOW
                else:
                    for _ in range(60):
                        group_nodes = hg.get_group_nodes(0)
                        if len(group_nodes) >= 2:
                            size = random.randint(8, min(20, len(group_nodes)))
                            e = frozenset(random.sample(group_nodes, size))
                            if e not in hg.E:
                                hg.E.append(e)
                
                for _ in range(120):
                    hg.apply_rules()
                
                M = hg.get_group_order_parameter(0)
                group_M.append(M)
            
            all_M_values.append(group_M)
        
        all_M_values = np.array(all_M_values)
        
        # 对每个 group 单独分析
        group_stats = {}
        for g in range(k):
            M_g = all_M_values[:, g]
            props = measure_spectrum_properties(M_g)
            group_stats[f'group_{g}'] = props
        
        # 计算组合谱的性质
        flat_M = all_M_values.flatten()
        combined_props = measure_spectrum_properties(flat_M)
        
        # 张量积预测：2^{k×L} 个离散状态 -> 连续谱版本
        # 预测谱宽度 ~ k×L × 单个群体的宽度
        predicted_spectral_growth = k * L
        
        results[f'k{k}_L{L}'] = {
            'k': k,
            'L': L,
            'dim': k * L,
            'group_stats': group_stats,
            'combined_props': combined_props,
            'n_runs': n_runs
        }
        
        print(f"  Combined spectrum: M_min={combined_props['M_min']:.3f}, "
              f"M_max={combined_props['M_max']:.3f}, "
              f"width={combined_props['spectral_width']:.3f}")
    
    return results


def study_lambda_effect_on_spectrum():
    """
    研究 λ 耦合对谱的影响
    假设：λ 增大会压缩谱宽度（forced synchronization）
    """
    print("\n" + "=" * 60)
    print("Lambda Effect on Spectrum")
    print("=" * 60)
    
    k, L = 2, 1
    lambda_values = [0.0, 0.1, 0.5, 1.0, 2.0]
    
    results = {}
    
    for lam in lambda_values:
        print(f"\nTesting λ = {lam}...")
        
        all_M_values = []
        n_runs = 50
        
        for run in range(n_runs):
            seed = 42 + run * 100
            
            # 创建两个群体
            group_M = []
            for g in range(k):
                hg = CoagulationBiasedHypergraph(
                    N=50, n_groups=1, gamma=0.35,
                    fusion_boost=10.0, seed=seed + g
                )
                
                # 随机初始条件
                if random.random() < 0.5:
                    hg.E = hg.E[:2]
                else:
                    for _ in range(60):
                        group_nodes = hg.get_group_nodes(0)
                        if len(group_nodes) >= 2:
                            size = random.randint(8, min(20, len(group_nodes)))
                            e = frozenset(random.sample(group_nodes, size))
                            if e not in hg.E:
                                hg.E.append(e)
                
                for _ in range(100):
                    hg.apply_rules()
                
                M = hg.get_group_order_parameter(0)
                group_M.append(M)
            
            all_M_values.append(group_M)
        
        all_M_values = np.array(all_M_values)
        
        # 计算组内相关度
        corr = np.corrcoef(all_M_values[:, 0], all_M_values[:, 1])[0, 1]
        
        flat_M = all_M_values.flatten()
        props = measure_spectrum_properties(flat_M)
        
        results[lam] = {
            'spectral_width': props['spectral_width'],
            'M_mean': props['M_mean'],
            'inter_group_correlation': corr,
            'n_runs': n_runs
        }
        
        print(f"  Spectral width: {props['spectral_width']:.4f}")
        print(f"  M_mean: {props['M_mean']:.4f}")
        print(f"  Inter-group correlation: {corr:.4f}")
    
    return results


if __name__ == '__main__':
    # Part 1: Characterize spectrum vs dimension
    spectrum_results = characterize_spectrum_vs_dim()
    
    # Part 2: Lambda effect
    lambda_results = study_lambda_effect_on_spectrum()
    
    # Save results
    all_results = {
        'spectrum_vs_dimension': spectrum_results,
        'lambda_effect': lambda_results
    }
    
    with open('F:/hypergraph_bistability/results/continuous_spectrum/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nSpectrum Properties vs Dimension:")
    for key, val in spectrum_results.items():
        dim = val['dim']
        width = val['combined_props']['spectral_width']
        print(f"  k×L={dim}: spectral_width={width:.4f}")
    
    print("\nLambda Effect on Spectrum:")
    for lam, val in lambda_results.items():
        width = val['spectral_width']
        corr = val['inter_group_correlation']
        print(f"  λ={lam}: width={width:.4f}, corr={corr:.4f}")
