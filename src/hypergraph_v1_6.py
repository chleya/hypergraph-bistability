"""
Hypergraph Evolution v1.6 - 度数惩罚版本
核心升级：软约束 hub 影响力，而非硬切断
影响 = 1 / (1 + degree)
"""

import numpy as np
import random

random.seed(42)
np.random.seed(42)

class Hypergraph_v1_6:
    def __init__(self, n_vertices=15):
        self.V = list(range(n_vertices))
        self.E = []
        self.s = {v: np.random.randn(4) for v in self.V}
        
        # 冲突图
        self.conflicts = set()
        
        # 阵营ID
        self.faction = {}
        
        # 统计
        self.rejection_count = 0
        self.scar_node = None
        
        # v1.6 参数：度数惩罚强度
        self.ALPHA = 2.0  # 惩罚系数
        
        # 初始化
        for _ in range(4):
            size = random.randint(2, 5)
            e = frozenset(random.sample(self.V, size))
            self.E.append(e)
            self.faction[e] = random.randint(0, 2)
    
    def get_avg_distance(self):
        if len(self.V) < 2: return 0.1
        samples = 30
        dist_sum = 0
        for _ in range(samples):
            u, v = random.sample(self.V, 2)
            dist_sum += np.linalg.norm(self.s[u] - self.s[v])
        return max(dist_sum / samples, 0.01)
    
    def get_node_degree(self, v):
        return sum(1 for e in self.E if v in e)
    
    def get_influence(self, v):
        """度数惩罚：影响力 = 1 / (1 + alpha * degree)"""
        degree = self.get_node_degree(v)
        return 1.0 / (1.0 + self.ALPHA * degree)
    
    def apply_rules(self):
        changed = False
        avg_dist = self.get_avg_distance()
        
        # 规则1: 支化 (Growth) - 【核心修改：度数惩罚】
        if random.random() < 0.35 and self.E:
            e = random.choice(self.E)
            
            # 按影响力加权选择节点
            nodes = list(e)
            weights = np.array([self.get_influence(v) for v in nodes])
            weights = weights / weights.sum()  # 归一化
            x = np.random.choice(nodes, p=weights)
            
            w = len(self.V)
            self.V.append(w)
            self.s[w] = self.s[x] + np.random.randn(4) * 0.2
            new_e = frozenset([x, w])
            self.E.append(new_e)
            self.faction[new_e] = self.faction.get(e, random.randint(0, 2))
            changed = True
        
        # 规则2: 融合 (Fusion) - 【核心修改：度数惩罚影响融合权重】
        if random.random() < 0.3 and len(self.E) >= 2:
            e1 = random.choice(self.E)
            e2 = random.choice(self.E)
            
            if e1 != e2 and len(e1 & e2) >= 1:
                faction1 = self.faction.get(e1, 0)
                faction2 = self.faction.get(e2, 0)
                faction_penalty = 0.5 if faction1 != faction2 else 0.0
                
                # 计算加权的状态中心
                weights1 = np.array([self.get_influence(v) for v in e1])
                weights1 = weights1 / weights1.sum()
                c1 = np.average([self.s[u] for u in e1], axis=0, weights=weights1)
                
                weights2 = np.array([self.get_influence(v) for v in e2])
                weights2 = weights2 / weights2.sum()
                c2 = np.average([self.s[u] for u in e2], axis=0, weights=weights2)
                
                distance = np.linalg.norm(c1 - c2)
                
                effective_threshold = 0.8 * avg_dist - faction_penalty
                
                if distance > effective_threshold:
                    self.rejection_count += 1
                    self.conflicts.add((e1, e2))
                    self.conflicts.add((e2, e1))
                    
                    # 排异导致状态分离
                    for v in e1:
                        self.s[v] += np.random.randn(4) * 0.3 * self.get_influence(v)
                    for v in e2:
                        self.s[v] -= np.random.randn(4) * 0.3 * self.get_influence(v)
                    
                    # 阵营分裂
                    if faction1 != faction2:
                        pass
                    else:
                        self.faction[e1] = faction1
                        self.faction[e2] = (faction1 + 1) % 3
                else:
                    new_e = frozenset(e1 | e2)
                    if new_e not in self.E:
                        self.E.remove(e1)
                        self.E.remove(e2)
                        self.E.append(new_e)
                        
                        # 状态扩散（也受度数惩罚影响）
                        shared = e1 & e2
                        if shared:
                            for v in shared:
                                influence = self.get_influence(v)
                                self.s[v] = (1 - influence) * self.s[v] + influence * np.mean([self.s[u] for u in shared], axis=0)
                        
                        self.faction[new_e] = (faction1 + faction2) % 3
                        changed = True
        
        # 规则3: 递归分裂
        if random.random() < 0.25 and self.E:
            e = random.choice(self.E)
            if len(e) >= 5:
                lst = list(e)
                mid = len(lst) // 2
                pivot = lst[mid]
                e_left = frozenset(lst[:mid] + [pivot])
                e_right = frozenset(lst[mid:])
                
                parent_faction = self.faction.get(e, 0)
                
                self.E.remove(e)
                self.E.append(e_left)
                self.E.append(e_right)
                
                self.faction[e_left] = parent_faction
                self.faction[e_right] = (parent_faction + 1) % 3
                changed = True
        
        # 规则4: 消除
        if random.random() < 0.15 and self.E:
            e = random.choice(self.E)
            if len(e) <= 2 and random.random() < 0.5:
                self.E.remove(e)
                if e in self.faction:
                    del self.faction[e]
                changed = True
        
        return changed
    
    def inject_you_pulse(self):
        """外部扰动：精准打击高影响力节点"""
        if not self.V: return
        
        # 优先打击高影响力节点
        influences = {v: self.get_influence(v) for v in self.V}
        # 按影响力排序，30% 概率打最高影响力，70% 打随机
        if random.random() < 0.3:
            target = max(influences, key=influences.get)
        else:
            target = random.choice(self.V)
        
        self.scar_node = target
        self.s[target] += np.array([5.0, -5.0, 5.0, -5.0]) 

def observe(H):
    n_edges = len(H.E)
    scar_degree = sum(1 for e in H.E if H.scar_node in e) if H.scar_node is not None else 0
    n_conflicts = len(H.conflicts) // 2
    
    faction_counts = {}
    for e in H.E:
        f = H.faction.get(e, 0)
        faction_counts[f] = faction_counts.get(f, 0) + 1
    
    avg_edge_size = np.mean([len(e) for e in H.E]) if H.E else 0
    
    # 统计度数分布
    degrees = [H.get_node_degree(v) for v in H.V]
    max_degree = max(degrees) if degrees else 0
    avg_degree = np.mean(degrees) if degrees else 0
    
    return n_edges, H.rejection_count, scar_degree, n_conflicts, faction_counts, avg_edge_size, max_degree, avg_degree

def run_evolution(steps=1000):
    H = Hypergraph_v1_6(n_vertices=15)
    print("--- 创世开始 (v1.6 度数惩罚版, alpha=2.0) ---")
    print("Step    | 超边数 | 排异 | 冲突 | 疤痕度 | 阵营分布 | 平均边 | 最大度")
    print("-" * 90)
    
    for t in range(1, steps + 1):
        H.apply_rules()
        
        if t % 50 == 0:
            H.inject_you_pulse()
        
        if t % 100 == 0:
            edges, rejects, scar_deg, conflicts, factions, avg_size, max_deg, avg_deg = observe(H)
            faction_str = "/".join([str(factions.get(i, 0)) for i in range(3)])
            print(f"{t:6d} | {edges:6d} | {rejects:4d} | {conflicts:4d} | {scar_deg:6d} | {faction_str} | {avg_size:.2f} | {max_deg:6.1f}")
    
    print("\n=== 最终状态 ===")
    edges, rejects, scar_deg, conflicts, factions, avg_size, max_deg, avg_deg = observe(H)
    print(f"超边数: {edges}")
    print(f"累计排异: {rejects}")
    print(f"冲突记忆: {conflicts}")
    print(f"疤痕节点度数: {scar_deg}")
    print(f"阵营分布: {factions}")
    print(f"平均超边大小: {avg_size:.2f}")
    print(f"最大度数: {max_deg:.1f}")
    print(f"平均度数: {avg_deg:.1f}")

if __name__ == "__main__":
    run_evolution(steps=1000)
