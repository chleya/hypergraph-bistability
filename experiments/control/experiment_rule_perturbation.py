"""
Rule Perturbation Experiment: Verify Structural Invariance
=========================================================

Tests whether H_core growth occurs across different rule probability configurations.
If H_core still grows when rules are perturbed, it proves the mechanism is robust
and not dependent on specific rule parameters.

Design:
- Baseline: Original probabilities (p_growth=0.3, p_fusion=0.25, p_split=0.12)
- Perturbed: Various combinations with modified probabilities
- Measure: ΔH_core = H_core(T) - H_core(0)
"""

import numpy as np
import random
from typing import Dict, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('F:/hypergraph_bistability/figures/reconstruction', exist_ok=True)


class HypergraphRulePerturb:
    """Hypergraph with configurable rule probabilities."""

    def __init__(
        self,
        N: int = 50,
        p_pair: float = 0.5,
        seed: int = 42,
        p_growth: float = 0.3,
        p_fusion: float = 0.25,
        p_split: float = 0.12
    ):
        random.seed(seed)
        np.random.seed(seed)

        self.N = N
        self.p_pair = p_pair
        self.K = int(0.35 * N)
        self.p_growth = p_growth
        self.p_fusion = p_fusion
        self.p_split = p_split

        self.V = list(range(N))
        self.E = []

        for _ in range(15):
            if random.random() < p_pair:
                e = frozenset(random.sample(self.V, 2))
            else:
                e = frozenset(random.sample(self.V, min(3, N)))
            self.E.append(e)

    def get_node_degree(self, v: int) -> int:
        return sum(1 for e in self.E if v in e)

    def get_clusters(self):
        if not self.V or not self.E:
            return [], set()

        adj = {v: set() for v in self.V}
        for e in self.E:
            nodes = list(e)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    adj[nodes[i]].add(nodes[j])
                    adj[nodes[j]].add(nodes[i])

        visited = set()
        clusters = []

        for s in self.V:
            if s in visited:
                continue
            stack = [s]
            c = set()
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                c.add(n)
                if n in adj:
                    for nb in adj[n]:
                        if nb not in visited:
                            stack.append(nb)
            clusters.append(c)

        max_cluster = max(clusters, key=len) if clusters else set()
        return clusters, max_cluster

    def compute_H(self, node_set):
        relevant_edges = [e for e in self.E if len(e & node_set) > 0]
        if not relevant_edges:
            return 0.0
        sizes = [len(e) for e in relevant_edges]
        counts = {}
        for s in sizes:
            counts[s] = counts.get(s, 0) + 1
        total = len(relevant_edges)
        H = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                H -= p * np.log2(p)
        return H

    def get_M(self):
        _, max_cluster = self.get_clusters()
        return len(max_cluster) / len(self.V) if max_cluster else 0.0

    def get_state(self):
        """Get H_core and M."""
        _, max_cluster = self.get_clusters()
        all_nodes = set(self.V)

        M = len(max_cluster) / len(self.V) if max_cluster else 0.0
        H_core = self.compute_H(max_cluster) if max_cluster else 0.0

        return M, H_core

    def apply_rules(self, steps: int = 1):
        for _ in range(steps):
            if random.random() < self.p_growth and self.E:
                e = random.choice(self.E)
                v = random.choice(list(e))
                w = max(self.V) + 1
                self.V.append(w)

                if random.random() < self.p_pair:
                    new_e = frozenset([v, w])
                else:
                    new_e = frozenset([v, w, random.choice(self.V[:-1])])
                self.E.append(new_e)

            if random.random() < self.p_fusion and len(self.E) >= 2:
                e1, e2 = random.sample(self.E, 2)
                if len(e1 & e2) >= 1 and random.random() < 0.5:
                    new_e = frozenset(e1 | e2)
                    if len(new_e) >= 2:
                        self.E.append(new_e)
                        if e1 in self.E:
                            self.E.remove(e1)
                        if e2 in self.E:
                            self.E.remove(e2)

            if random.random() < self.p_split:
                large = [e for e in self.E if len(e) > 2]
                if large:
                    e = random.choice(large)
                    nodes = list(e)
                    if len(nodes) >= 4:
                        split = len(nodes) // 2
                        self.E.append(frozenset(nodes[:split]))
                        self.E.append(frozenset(nodes[split:]))
                        self.E.remove(e)

            for v in list(self.V):
                d = self.get_node_degree(v)
                if d > self.K:
                    excess = d - self.K
                    v_edges = [e for e in self.E if v in e]
                    for e in v_edges[:excess]:
                        if len(e) > 2:
                            new_e = e - {v}
                            if len(new_e) >= 2:
                                self.E.remove(e)
                                self.E.append(new_e)


def run_rule_perturbation(
    n_runs: int = 5,
    T: int = 80
) -> Dict:
    """Run rule perturbation experiment."""

    print("=" * 70)
    print("Rule Perturbation Experiment: Structural Invariance")
    print("=" * 70)

    configs = [
        {
            'name': 'Baseline',
            'p_growth': 0.30,
            'p_fusion': 0.25,
            'p_split': 0.12,
            'description': 'Original probabilities'
        },
        {
            'name': 'No Fusion',
            'p_growth': 0.30,
            'p_fusion': 0.00,
            'p_split': 0.12,
            'description': 'Fusion disabled'
        },
        {
            'name': 'No Split',
            'p_growth': 0.30,
            'p_fusion': 0.25,
            'p_split': 0.00,
            'description': 'Split disabled'
        },
        {
            'name': 'High Growth',
            'p_growth': 0.50,
            'p_fusion': 0.10,
            'p_split': 0.10,
            'description': 'More growth, less fusion'
        },
        {
            'name': 'High Split',
            'p_growth': 0.20,
            'p_fusion': 0.15,
            'p_split': 0.30,
            'description': 'More split, less growth'
        },
        {
            'name': 'All Equal',
            'p_growth': 0.25,
            'p_fusion': 0.25,
            'p_split': 0.25,
            'description': 'Equal probabilities'
        },
        {
            'name': 'Low Activity',
            'p_growth': 0.10,
            'p_fusion': 0.05,
            'p_split': 0.05,
            'description': 'Reduced overall activity'
        },
        {
            'name': 'High Fusion',
            'p_growth': 0.15,
            'p_fusion': 0.50,
            'p_split': 0.05,
            'description': 'Heavy fusion'
        },
    ]

    results = {}

    for config in configs:
        print(f"\n[{config['name']}] {config['description']}")
        print(f"  p_growth={config['p_growth']}, p_fusion={config['p_fusion']}, p_split={config['p_split']}")

        delta_H_list = []
        delta_M_list = []
        H_final_list = []
        M_final_list = []

        for run in range(n_runs):
            h = HypergraphRulePerturb(
                N=50,
                p_pair=0.5,
                seed=run * 100 + 42,
                p_growth=config['p_growth'],
                p_fusion=config['p_fusion'],
                p_split=config['p_split']
            )

            _, max_cluster = h.get_clusters()
            H_initial = h.compute_H(max_cluster) if max_cluster else 0.0
            M_initial = h.get_M()

            for t in range(T):
                h.apply_rules()

            M_final, H_final = h.get_state()
            delta_H = H_final - H_initial
            delta_M = M_final - M_initial

            delta_H_list.append(delta_H)
            delta_M_list.append(delta_M)
            H_final_list.append(H_final)
            M_final_list.append(M_final)

        results[config['name']] = {
            'delta_H_mean': np.mean(delta_H_list),
            'delta_H_std': np.std(delta_H_list),
            'delta_M_mean': np.mean(delta_M_list),
            'delta_M_std': np.std(delta_M_list),
            'H_final_mean': np.mean(H_final_list),
            'M_final_mean': np.mean(M_final_list),
            'config': config
        }

        print(f"  ΔH_core = {np.mean(delta_H_list):+.3f} ± {np.std(delta_H_list):.3f}")
        print(f"  ΔM = {np.mean(delta_M_list):+.3f} ± {np.std(delta_M_list):.3f}")

    return results


def plot_rule_perturbation_results(results: Dict):
    """Plot rule perturbation results."""

    names = list(results.keys())
    delta_H_means = [results[n]['delta_H_mean'] for n in names]
    delta_H_stds = [results[n]['delta_H_std'] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    colors = ['green' if dH > 0 else 'red' for dH in delta_H_means]
    bars = ax1.bar(range(len(names)), delta_H_means, yerr=delta_H_stds,
                   color=colors, alpha=0.7, capsize=5, edgecolor='black')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('ΔH_core = H_core(T) - H_core(0)')
    ax1.set_title('H_core Growth Across Rule Configurations')
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, dH in zip(bars, delta_H_means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{dH:+.2f}', ha='center', va='bottom', fontsize=9)

    growth_rates = []
    for name in names:
        cfg = results[name]['config']
        p_active = cfg['p_growth'] + cfg['p_fusion'] + cfg['p_split']
        growth_rate = results[name]['delta_H_mean'] / max(p_active, 0.01)
        growth_rates.append(growth_rate)

    ax2 = axes[1]
    ax2.bar(range(len(names)), growth_rates, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Growth Efficiency (ΔH / Total Activity)')
    ax2.set_title('Growth Efficiency Across Configurations')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    save_path = 'F:/hypergraph_bistability/figures/reconstruction/rule_perturbation.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[Saved] {save_path}")

    plt.close()

    return save_path


def print_summary(results: Dict):
    """Print summary of results."""

    print("\n" + "=" * 70)
    print("SUMMARY: Rule Perturbation Results")
    print("=" * 70)

    print("\n[Table: H_core Growth Across Configurations]")
    print("-" * 70)
    print(f"{'Config':<15} {'ΔH_core':<15} {'ΔM':<15} {'H_present?':<10}")
    print("-" * 70)

    h_core_grows_count = 0
    for name, res in results.items():
        delta_H = res['delta_H_mean']
        delta_M = res['delta_M_mean']
        grows = "YES" if delta_H > 0 else "NO"
        if delta_H > 0:
            h_core_grows_count += 1
        print(f"{name:<15} {delta_H:+.3f} ± {res['delta_H_std']:.3f}   "
              f"{delta_M:+.3f} ± {res['delta_M_std']:.3f}   {grows:<10}")

    print("-" * 70)

    print(f"\n[Conclusion]")
    print(f"  H_core grew in {h_core_grows_count}/{len(results)} configurations")
    print(f"  ")

    if h_core_grows_count == len(results):
        print(f"  => Structural Invariance CONFIRMED")
        print(f"  => H_core growth is robust across ALL rule configurations")
        print(f"  => In-situ mechanism does not depend on specific rule probabilities")
    elif h_core_grows_count > len(results) / 2:
        print(f"  => Partial Structural Invariance")
        print(f"  => H_core growth occurs in majority of configurations")
    else:
        print(f"  => Structural Invariance NOT confirmed")
        print(f"  => H_core growth depends on specific rule combinations")


if __name__ == '__main__':
    results = run_rule_perturbation(n_runs=5, T=80)
    plot_rule_perturbation_results(results)
    print_summary(results)
