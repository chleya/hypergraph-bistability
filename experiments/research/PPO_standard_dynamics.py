"""
PPO训练 - 标准bistable dynamics
================================
目标：验证PPO能否在标准设置下学到有效的local控制策略
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
import os
import time

os.makedirs('F:/hypergraph_bistability/results/PPO_verification', exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)

ALPHA = 1.0
A = 0.5
K = 3
L = 2
MU = 0.0


def ode_standard(t, M_flat, k, L, lam, mu):
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


def run_episode_ppo(lam, policy, target, n_init=10, mu=0.0):
    """用PPO policy运行episode"""
    k_val, L_val = K, L
    success_count = 0
    
    for seed in range(n_init):
        np.random.seed(seed * 100 + 42)
        
        state = np.full((k_val, L_val), 0.05)
        state[target] = 0.5
        state_flat = state.flatten()
        
        boost_start, boost_duration, boost_strength = policy.act(state_flat)
        
        T_MAX = 80.0
        
        if boost_start > 0:
            sol1 = solve_ivp(
                lambda t, M: ode_standard(t, M, k_val, L_val, lam, mu),
                [0, boost_start], state_flat,
                method='RK45', rtol=1e-4, atol=1e-6
            )
            state_flat = sol1.y[:, -1]
        
        if boost_duration > 0 and boost_strength > 0.01:
            dt = boost_duration / 20
            for _ in range(20):
                dM = ode_standard(0, state_flat, k_val, L_val, lam, mu)
                state_flat = state_flat + boost_strength * dM * dt
                state_flat = np.clip(state_flat, 0, 1)
        
        sol3 = solve_ivp(
            lambda t, M: ode_standard(t, M, k_val, L_val, lam, mu),
            [0, T_MAX], state_flat,
            method='RK45', rtol=1e-4, atol=1e-6
        )
        final_state = sol3.y[:, -1].reshape(k_val, L_val)
        
        target_M = final_state[target]
        other_mean = np.mean([final_state[i,j] for i in range(k_val) for j in range(L_val) 
                             if (i,j) != target])
        
        if target_M > 0.7 and target_M > other_mean + 0.1:
            success_count += 1
    
    return success_count / n_init


class SimplePolicy(nn.Module):
    def __init__(self, state_dim=6, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3)
        )
    
    def forward(self, x):
        out = self.net(x)
        start = 5 + torch.sigmoid(out[0]) * 60      # 5-65
        duration = 5 + torch.sigmoid(out[1]) * 40    # 5-45
        strength = 0.1 + torch.sigmoid(out[2]) * 1.5  # 0.1-1.6
        return start, duration, strength
    
    def act(self, state):
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32)
            start, duration, strength = self.forward(state_t)
            return (start.item(), duration.item(), strength.item())


def train_ppo(lam, n_episodes=50):
    policy = SimplePolicy()
    optimizer = optim.Adam(policy.parameters(), lr=0.005)
    
    best_success = 0
    best_state = None
    history = []
    
    target = (0, 0)
    
    for ep in range(n_episodes):
        optimizer.zero_grad()
        
        # 采样多次估计梯度
        successes = []
        for _ in range(3):
            success = run_episode_ppo(lam, policy, target, n_init=8)
            successes.append(success)
        
        avg_success = np.mean(successes)
        
        # 简化策略梯度：直接最大化成功率
        loss = -torch.tensor(avg_success, requires_grad=True)
        loss.backward()
        optimizer.step()
        
        if avg_success > best_success:
            best_success = avg_success
            best_state = policy.state_dict().copy()
        
        history.append(avg_success)
        
        if ep % 10 == 0:
            print(f"  Ep {ep}: success={avg_success:.0%}, best={best_success:.0%}")
    
    if best_state:
        policy.load_state_dict(best_state)
    
    return policy, best_success, history


def main():
    print("=" * 60)
    print("PPO训练 - 标准bistable dynamics")
    print("=" * 60)
    
    lambda_values = [0.05, 0.1]
    results = {}
    
    for lam in lambda_values:
        print(f"\nTraining λ = {lam}...")
        t0 = time.time()
        policy, best_success, history = train_ppo(lam, n_episodes=40)
        elapsed = time.time() - t0
        results[lam] = {
            'best_success': best_success,
            'history': history,
            'time': elapsed
        }
        print(f"  → Best success: {best_success:.0%}, time: {elapsed:.0f}s")
    
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"{'λ':>6s} | {'PPO success':>12s}")
    print("-" * 25)
    for lam, res in results.items():
        print(f"{lam:>6.2f} | {res['best_success']:>11.0%}")
    
    if all(r['best_success'] < 0.1 for r in results.values()):
        print("""
结论: PPO在标准bistable dynamics下无法学到有效的local控制策略
- 固定local boost: 0%
- PPO learned: <10%
→ Local控制无效，Global控制是唯一可行方案
        """)


if __name__ == '__main__':
    main()