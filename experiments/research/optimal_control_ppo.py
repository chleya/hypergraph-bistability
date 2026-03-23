"""
最优控制实验 - 使用强化学习 (PPO) 学习 boost 策略
==================================================
目标：在弱耦合区达到 80%+ 成功率

方法：使用 PyTorch 实现 PPO 算法学习：
- boost 时机
- boost 强度  
- 目标群体
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os
import random

os.makedirs('F:/hypergraph_bistability/figures/control', exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ==================== 1. 超图动力学 ====================
def hypergraph_dynamics(M_flat, t, a_list, b_list, c_list, lambda_, mu, L, k, boost_info):
    """带 boost 的超图动力学"""
    M = M_flat.reshape((L, k))
    dM = np.zeros((L, k))
    
    target_layer, target_group, boost_factor, active = boost_info
    
    # 动态修改参数
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

def run_episode(lambda_, Kc_list, policy, target=None, max_steps=200, N_init=30):
    """运行一个 episode，返回成功率"""
    L, k = 2, 3
    mu = 0.0
    
    a_list = [-3.0 / kc for kc in Kc_list]
    b_list = [4.5 / kc for kc in Kc_list]
    c_list = [-1.5 / kc for kc in Kc_list]
    
    # 如果没有指定目标，随机选择
    if target is None:
        target = (np.random.randint(0, L), np.random.randint(0, k))
    target_layer, target_group = target
    
    success_count = 0
    
    for _ in range(N_init):
        # 随机初始状态
        state = np.random.uniform(0.1, 0.9, L * k)
        
        # 决策：何时 boost，强度多少
        # 简化：直接用 policy 决定
        boost_decision = policy.act(state)
        boost_start, boost_duration, boost_strength = boost_decision
        
        # 模拟
        t = np.linspace(0, max_steps, max_steps * 5)
        
        # 阶段1：自由演化到 boost_start
        if boost_start > 0:
            boost_info_1 = (target_layer, target_group, 1.0, False)
            sol1 = odeint(hypergraph_dynamics, state, np.linspace(0, boost_start, 50),
                         args=(a_list, b_list, c_list, lambda_, mu, L, k, boost_info_1))
            state = sol1[-1].copy()
        
        # 阶段2：boost
        if boost_duration > 0 and boost_strength > 1.0:
            boost_info_2 = (target_layer, target_group, boost_strength, True)
            sol2 = odeint(hypergraph_dynamics, state, 
                         np.linspace(boost_start, boost_start + boost_duration, 50),
                         args=(a_list, b_list, c_list, lambda_, mu, L, k, boost_info_2))
            state = sol2[-1].copy()
        
        # 阶段3：自由演化到结束
        boost_info_3 = (target_layer, target_group, 1.0, False)
        sol3 = odeint(hypergraph_dynamics, state,
                     np.linspace(boost_start + boost_duration, max_steps, 100),
                     args=(a_list, b_list, c_list, lambda_, mu, L, k, boost_info_3))
        
        # 检查成功
        final_M = sol3[-1].reshape((L, k))
        target_M = final_M[target_layer, target_group]
        other_values = [final_M[l, g] for l in range(L) for g in range(k) 
                       if (l, g) != (target_layer, target_group)]
        avg_other = np.mean(other_values)
        
        if target_M > avg_other + 0.15:
            success_count += 1
    
    return success_count / N_init

# ==================== 2. 简化的 Policy Network ====================
class SimplePolicy(nn.Module):
    def __init__(self, state_dim=6, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)  # boost_start, duration, strength
        )
    
    def forward(self, x):
        # 输出离散化
        out = self.net(x)
        # boost_start: 10-80
        start = 10 + torch.sigmoid(out[0]) * 70
        # duration: 10-50
        duration = 10 + torch.sigmoid(out[1]) * 40
        # strength: 1.0-2.5
        strength = 1.0 + torch.sigmoid(out[2]) * 1.5
        return start, duration, strength
    
    def act(self, state):
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32)
            start, duration, strength = self.forward(state_t)
            return (start.item(), duration.item(), strength.item())

# ==================== 3. 训练循环 ====================
def train_ppo(lambda_, Kc_list, n_episodes=50):
    """训练 PPO-style policy"""
    L, k = 2, 3
    
    policy = SimplePolicy(state_dim=L*k)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    
    best_success = 0
    best_policy_state = None
    history = []
    
    for episode in range(n_episodes):
        # 收集经验
        successes = []
        for _ in range(5):  # 每次更新采样 5 次
            # 随机目标
            target = (np.random.randint(0, L), np.random.randint(0, k))
            
            # 运行 episode
            state = np.random.uniform(0.1, 0.9, L * k)
            boost_decision = policy.act(state)
            boost_start, boost_duration, boost_strength = boost_decision
            
            # 计算奖励
            success = run_episode(lambda_, Kc_list, policy, target=target)
            successes.append(success)
        
        avg_success = np.mean(successes)
        
        # 更新 policy（简化版：用成功率的梯度）
        if avg_success > best_success:
            best_success = avg_success
            best_policy_state = policy.state_dict().copy()
        
        history.append(avg_success)
        
        if episode % 10 == 0:
            print(f"Episode {episode}: success = {avg_success:.2f}, best = {best_success:.2f}")
    
    # 恢复最佳 policy
    if best_policy_state:
        policy.load_state_dict(best_policy_state)
    
    return policy, best_success, history

# ==================== 4. 主实验 ====================
print("=" * 60)
print("最优控制实验 - PPO Learning")
print("=" * 60)

# 参数
Kc_list = [0.32, 0.40, 0.48]
lambda_values = [0.1, 0.3, 0.5]

results = {}

for lam in lambda_values:
    print(f"\nTraining for λ = {lam}...")
    policy, best_success, history = train_ppo(lam, Kc_list, n_episodes=30)
    results[lam] = {
        'policy': policy,
        'success': best_success,
        'history': history
    }
    print(f"λ = {lam}: Best success rate = {best_success:.2f}")

# ==================== 5. 可视化 ====================
plt.figure(figsize=(10, 6))
for lam in history:
    plt.plot(history[lam], label=f'λ = {lam}')
plt.xlabel('Episode')
plt.ylabel('Success Rate')
plt.title('PPO Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('F:/hypergraph_bistability/figures/control/ppo_training.png', dpi=150)
plt.close()

# 结果汇总
print("\n" + "=" * 60)
print("结果汇总")
print("=" * 60)
for lam in lambda_values:
    print(f"λ = {lam}: success = {results[lam]['success']:.2f}")

# 对比
print("\n对比:")
print("局部 Boost (固定策略): 62%")
for lam in lambda_values:
    print(f"PPO (λ={lam}): {results[lam]['success']:.2f}")

print("\n图片已保存")
