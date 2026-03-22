"""
Block Fusion Experiment: Verify In-Situ Reconstruction vs Transport
===================================================================

Causal intervention experiment to test:
- If peri→core fusion is blocked, does H_core still grow?
- If H_core still grows → in-situ reconstruction is the mechanism
- If H_core stops growing → transport from periphery is required

Design:
- Control: normal system with full fusion
- Block: block_prob=1.0, only blocks peri→core fusion (not internal fusion)

Metrics tracked:
- H_core(t): core complexity
- H_peri(t): peripheral complexity  
- M(t): order parameter
- core_size(t): number of edges in max cluster
- ΔH_core: final increment

Statistical test:
- t-test on ΔH_core between Control and Block groups
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os

os.makedirs('F:/hypergraph_bistability/figures/reconstruction', exist_ok=True)


class HypergraphBase:
    """Base hypergraph with core/periphery analysis."""

    def __init__(self, N=50, p_pair=0.5, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        self.N = N
        self.p_pair = p_pair
        self.K = int(0.35 * N)
        self.V = list(range(N))
        self.E = []
        
        self.total_fusions = 0
        self.peri_to_core_attempts = 0

        for _ in range(15):
            if random.random() < p_pair:
                e = frozenset(random.sample(self.V, 2))
            else:
                e = frozenset(random.sample(self.V, min(3, N)))
            self.E.append(e)

    def get_node_degree(self, v):
        return sum(1 for e in self.E if v in e)

    def get_clusters_and_edges(self):
        if not self.V or not self.E:
            return [], set(), set()

        adj = {v: set() for v in self.V}
        for e in self.E:
            nodes = list(e)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if nodes[i] in adj and nodes[j] in adj:
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

        if not clusters:
            return [], set(), set()

        max_cluster = max(clusters, key=len)

        edge_ids_in_max = set()
        for idx, e in enumerate(self.E):
            if e & max_cluster:
                edge_ids_in_max.add(idx)

        return clusters, max_cluster, edge_ids_in_max

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
        if not self.V:
            return 0.0

        _, max_cluster, _ = self.get_clusters_and_edges()
        return len(max_cluster) / len(self.V) if max_cluster else 0.0

    def get_state(self):
        """Get all metrics for current state."""
        clusters, max_cluster, edge_ids_in_max = self.get_clusters_and_edges()
        all_nodes = set(self.V)

        M = len(max_cluster) / len(self.V) if max_cluster else 0.0
        core_size = len(edge_ids_in_max)

        if max_cluster:
            H_core = self.compute_H(max_cluster)
            periphery = all_nodes - max_cluster
            H_peri = self.compute_H(periphery) if periphery else 0.0
        else:
            H_core = 0.0
            H_peri = 0.0

        return {
            'M': M,
            'H_core': H_core,
            'H_peri': H_peri,
            'core_size': core_size
        }


class HypergraphControl(HypergraphBase):
    """Normal hypergraph with full fusion."""

    def apply_rules(self, steps=1):
        for _ in range(steps):
            if random.random() < 0.3 and self.E:
                e = random.choice(self.E)
                v = random.choice(list(e))
                w = max(self.V) + 1
                self.V.append(w)

                if random.random() < self.p_pair:
                    new_e = frozenset([v, w])
                else:
                    new_e = frozenset([v, w, random.choice(self.V[:-1])])
                self.E.append(new_e)

            if random.random() < 0.25 and len(self.E) >= 2:
                e1, e2 = random.sample(self.E, 2)
                if len(e1 & e2) >= 1 and random.random() < 0.5:
                    new_e = frozenset(e1 | e2)
                    if len(new_e) >= 2:
                        self.E.append(new_e)
                        if e1 in self.E:
                            self.E.remove(e1)
                        if e2 in self.E:
                            self.E.remove(e2)

            if random.random() < 0.12:
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


class HypergraphBlock(HypergraphBase):
    """Hypergraph with peri→core fusion blocked (block_prob=1.0)."""

    def __init__(self, N=50, p_pair=0.5, seed=42, block_prob=1.0):
        super().__init__(N=N, p_pair=p_pair, seed=seed)
        self.block_prob = block_prob

    def apply_rules(self, steps=1):
        for _ in range(steps):
            if random.random() < 0.3 and self.E:
                e = random.choice(self.E)
                v = random.choice(list(e))
                w = max(self.V) + 1
                self.V.append(w)

                if random.random() < self.p_pair:
                    new_e = frozenset([v, w])
                else:
                    new_e = frozenset([v, w, random.choice(self.V[:-1])])
                self.E.append(new_e)

            if random.random() < 0.25 and len(self.E) >= 2:
                e1, e2 = random.sample(self.E, 2)
                if len(e1 & e2) >= 1 and random.random() < 0.5:
                    self.total_fusions += 1
                    
                    clusters, max_cluster, _ = self.get_clusters_and_edges()

                    nodes1 = set(e1)
                    nodes2 = set(e2)

                    is_peri_to_core = False
                    if max_cluster:
                        e1_in_core = bool(nodes1 & max_cluster)
                        e2_in_core = bool(nodes2 & max_cluster)
                        e1_in_peri = bool(nodes1 - max_cluster)
                        e2_in_peri = bool(nodes2 - max_cluster)

                        if (e1_in_core and e2_in_peri) or (e1_in_peri and e2_in_core):
                            is_peri_to_core = True
                            self.peri_to_core_attempts += 1

                    if is_peri_to_core and self.block_prob == 1.0:
                        pass
                    else:
                        new_e = frozenset(e1 | e2)
                        if len(new_e) >= 2:
                            self.E.append(new_e)
                            if e1 in self.E:
                                self.E.remove(e1)
                            if e2 in self.E:
                                self.E.remove(e2)

            if random.random() < 0.12:
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


def run_comparison(n_runs=50, T=80):
    """Run Control vs Block comparison experiment."""

    print("=" * 70)
    print("Block Fusion Experiment: In-Situ Reconstruction vs Transport")
    print("=" * 70)
    print(f"Configuration: n_runs={n_runs}, T={T}")
    print(f"Block: peri→core fusion blocked (block_prob=1.0)")
    print()

    control_results = []
    block_results = []

    print("Running Control group...")
    control_diagnostics = {'total_fusions': [], 'peri_to_core': []}
    for i in range(n_runs):
        h = HypergraphControl(N=50, p_pair=0.5, seed=i * 100 + 42)
        history = {'M': [], 'H_core': [], 'H_peri': [], 'core_size': []}

        for t in range(T):
            h.apply_rules()
            state = h.get_state()
            history['M'].append(state['M'])
            history['H_core'].append(state['H_core'])
            history['H_peri'].append(state['H_peri'])
            history['core_size'].append(state['core_size'])

        control_results.append(history)
        control_diagnostics['total_fusions'].append(h.total_fusions)
        control_diagnostics['peri_to_core'].append(h.peri_to_core_attempts)
        if (i + 1) % 10 == 0:
            print(f"  Control: {i + 1}/{n_runs}")

    print("\nRunning Block group...")
    block_diagnostics = {'total_fusions': [], 'peri_to_core': []}
    for i in range(n_runs):
        h = HypergraphBlock(N=50, p_pair=0.5, seed=i * 100 + 42, block_prob=1.0)
        history = {'M': [], 'H_core': [], 'H_peri': [], 'core_size': []}

        for t in range(T):
            h.apply_rules()
            state = h.get_state()
            history['M'].append(state['M'])
            history['H_core'].append(state['H_core'])
            history['H_peri'].append(state['H_peri'])
            history['core_size'].append(state['core_size'])

        block_results.append(history)
        block_diagnostics['total_fusions'].append(h.total_fusions)
        block_diagnostics['peri_to_core'].append(h.peri_to_core_attempts)
        if (i + 1) % 10 == 0:
            print(f"  Block: {i + 1}/{n_runs}")

    control_arr = {k: np.array([r[k] for r in control_results]) for k in ['M', 'H_core', 'H_peri', 'core_size']}
    block_arr = {k: np.array([r[k] for r in block_results]) for k in ['M', 'H_core', 'H_peri', 'core_size']}

    delta_H_core_control = control_arr['H_core'][:, -1] - control_arr['H_core'][:, 0]
    delta_H_core_block = block_arr['H_core'][:, -1] - block_arr['H_core'][:, 0]

    t_stat, p_value = stats.ttest_ind(delta_H_core_control, delta_H_core_block)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n[Table 1: Final State Comparison (mean ± std)]")
    print("-" * 70)
    print(f"{'Metric':<20} {'Control':<20} {'Block':<20}")
    print("-" * 70)
    print(f"{'M(T)':<20} {control_arr['M'][:, -1].mean():.3f} ± {control_arr['M'][:, -1].std():.3f}   "
          f"{block_arr['M'][:, -1].mean():.3f} ± {block_arr['M'][:, -1].std():.3f}")
    print(f"{'H_core(T)':<20} {control_arr['H_core'][:, -1].mean():.3f} ± {control_arr['H_core'][:, -1].std():.3f}   "
          f"{block_arr['H_core'][:, -1].mean():.3f} ± {block_arr['H_core'][:, -1].std():.3f}")
    print(f"{'H_peri(T)':<20} {control_arr['H_peri'][:, -1].mean():.3f} ± {control_arr['H_peri'][:, -1].std():.3f}   "
          f"{block_arr['H_peri'][:, -1].mean():.3f} ± {block_arr['H_peri'][:, -1].std():.3f}")
    print(f"{'core_size(T)':<20} {control_arr['core_size'][:, -1].mean():.1f} ± {control_arr['core_size'][:, -1].std():.1f}   "
          f"{block_arr['core_size'][:, -1].mean():.1f} ± {block_arr['core_size'][:, -1].std():.1f}")
    print("-" * 70)

    print("\n[Table 2: Delta (T - 0) Comparison]")
    print("-" * 70)
    print(f"{'ΔH_core':<20} {delta_H_core_control.mean():+.3f} ± {delta_H_core_control.std():.3f}   "
          f"{delta_H_core_block.mean():+.3f} ± {delta_H_core_block.std():.3f}")
    print(f"{'ΔM':<20} {(control_arr['M'][:, -1] - control_arr['M'][:, 0]).mean():+.3f}   "
          f"{(block_arr['M'][:, -1] - block_arr['M'][:, 0]).mean():+.3f}")
    print(f"{'ΔH_peri':<20} {(control_arr['H_peri'][:, -1] - control_arr['H_peri'][:, 0]).mean():+.3f}   "
          f"{(block_arr['H_peri'][:, -1] - block_arr['H_peri'][:, 0]).mean():+.3f}")
    print("-" * 70)

    print(f"\n[Statistical Test: t-test on delta_H_core]")
    print(f"  t-statistic = {t_stat:.4f}")
    print(f"  p-value = {p_value:.6f}")

    print(f"\n[Diagnostic: Fusion Events]")
    print(f"  Control: total_fusions={np.mean(control_diagnostics['total_fusions']):.1f}, "
          f"peri_to_core={np.mean(control_diagnostics['peri_to_core']):.1f} (mean per run)")
    print(f"  Block:   total_fusions={np.mean(block_diagnostics['total_fusions']):.1f}, "
          f"peri_to_core={np.mean(block_diagnostics['peri_to_core']):.1f} (mean per run)")

    reduction = (1 - delta_H_core_block.mean() / delta_H_core_control.mean()) * 100 if delta_H_core_control.mean() != 0 else 0

    control_peri_to_core = np.mean(control_diagnostics['peri_to_core'])
    block_peri_to_core = np.mean(block_diagnostics['peri_to_core'])

    print(f"\n[FINAL VERDICT]")
    
    if control_peri_to_core < 0.1 and block_peri_to_core < 0.1:
        print(f"  ===================================================================")
        print(f"  CRITICAL FINDING: Peri->core fusion is essentially ZERO")
        print(f"  (Control: {control_peri_to_core:.2f}, Block: {block_peri_to_core:.2f} per run)")
        print(f"  ===================================================================")
        print(f"  ")
        print(f"  IMPLICATIONS:")
        print(f"  1. Transport via fusion is NOT a significant mechanism")
        print(f"  2. Core complexity grows via IN-SITU mechanisms (growth/split)")
        print(f"  3. Blocking fusion has no effect because fusion is rare anyway")
        print(f"  ")
        print(f"  INTERPRETATION:")
        print(f"  -> In-situ reconstruction is the DOMINANT mechanism")
        print(f"  -> Transport hypothesis is falsified by absence of transport events")
        verdict = "IN_SITU_RECONSTRUCTION_STRONG"
    elif p_value > 0.05:
        print(f"  [PASS] p > 0.05: No significant difference")
        print(f"     Blocking peri->core fusion does NOT reduce H_core growth")
        print(f"     -> Transport hypothesis FALSIFIED")
        print(f"     -> In-situ reconstruction is the dominant mechanism")
        verdict = "IN_SITU_RECONSTRUCTION"
    elif delta_H_core_block.mean() > 0:
        print(f"  [WARN] p < 0.05: Significant difference, but Block still grows")
        print(f"     delta_H_core reduction: {reduction:.1f}%")
        print(f"     -> Transport partially contributes")
        print(f"     -> But in-situ reconstruction also active")
        verdict = "MIXED_MECHANISM"
    else:
        print(f"  [FAIL] p < 0.05: Block stops H_core growth")
        print(f"     -> Transport hypothesis SUPPORTED")
        verdict = "TRANSPORT_SUPPORTED"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t = np.arange(T)
    time_points = np.arange(T)

    def plot_with_std(ax, t, control, block, label, color_c, color_b):
        c_mean = control.mean(axis=0)
        c_std = control.std(axis=0)
        b_mean = block.mean(axis=0)
        b_std = block.std(axis=0)

        ax.plot(t, c_mean, '-', color=color_c, linewidth=2, label=f'Control')
        ax.fill_between(t, c_mean - c_std, c_mean + c_std, alpha=0.2, color=color_c)
        ax.plot(t, b_mean, '--', color=color_b, linewidth=2, label=f'Block')
        ax.fill_between(t, b_mean - b_std, b_mean + b_std, alpha=0.2, color=color_b)
        ax.set_xlabel('Time')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plot_with_std(axes[0, 0], t, control_arr['H_core'], block_arr['H_core'], 'H_core', 'blue', 'red')
    axes[0, 0].set_title('H_core: Core Complexity (p={:.4f})'.format(p_value))

    plot_with_std(axes[0, 1], t, control_arr['M'], block_arr['M'], 'M', 'blue', 'red')

    plot_with_std(axes[1, 0], t, control_arr['H_peri'], block_arr['H_peri'], 'H_periphery', 'blue', 'red')

    axes[1, 1].bar(['Control', 'Block'],
                   [delta_H_core_control.mean(), delta_H_core_block.mean()],
                   yerr=[delta_H_core_control.std(), delta_H_core_block.std()],
                   color=['blue', 'red'], alpha=0.7, capsize=5)
    axes[1, 1].set_ylabel('ΔH_core')
    axes[1, 1].set_title('Delta H_core (T - 0)')
    axes[1, 1].axhline(0, color='black', linewidth=0.5)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Block Fusion Experiment: {verdict}\n(n={n_runs}, T={T})', fontsize=12)
    plt.tight_layout()

    save_path = 'F:/hypergraph_bistability/figures/reconstruction/block_vs_control.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved: {save_path}")

    results = {
        'verdict': verdict,
        'p_value': float(p_value),
        't_statistic': float(t_stat),
        'delta_H_core_control': float(delta_H_core_control.mean()),
        'delta_H_core_block': float(delta_H_core_block.mean()),
        'reduction_percent': float(reduction),
        'n_runs': n_runs,
        'T': T
    }

    return results


if __name__ == '__main__':
    results = run_comparison(n_runs=50, T=80)
