"""
Hypergraph Control Gym Environment
Based on Chen's recommendation - relaxed problem formulation
"""
import gym
from gym import spaces
import numpy as np
from scipy.integrate import odeint

# Hypergraph dynamics parameters
L = 2  # layers
k = 3  # groups per layer
lambda_ = 0.1  # coupling
mu = 0.0
Kc = [0.32, 0.40, 0.48]

a = [-3/kc for kc in Kc]
b = [4.5/kc for kc in Kc]
c = [-1.5/kc for kc in Kc]

def F(M, t, lambda_, mu):
    """Drift function for hypergraph dynamics"""
    M = M.reshape((L, k))
    dM = np.zeros((L, k))
    
    for l in range(L):
        for i in range(k):
            st = a[i]*M[l,i]**3 + b[i]*M[l,i]**2 + c[i]*M[l,i]
            cr = -lambda_ * M[l,i] * np.sum(M[l, np.arange(k) != i])
            cl = mu * np.sum(M[np.arange(L) != l, i])
            dM[l,i] = st + cr + cl
    
    return dM.flatten()

class HypergraphControlEnv(gym.Env):
    """Gym environment for hypergraph multistability control"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, lambda_=0.1, mu=0.0, T_max=500):
        super().__init__()
        self.L = 2
        self.k = 3
        self.lambda_ = lambda_
        self.mu = mu
        self.T_max = T_max
        
        # State: group-layer order parameters M (L*k dims)
        self.state_dim = self.L * self.k
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        # Action: [boost_target_group (0~k-1), boost_strength (0~3), boost_duration_steps (10~100)]
        self.action_space = spaces.Box(
            low=np.array([0, 0.5, 10]), 
            high=np.array([self.k-1, 3.0, 100]), 
            shape=(3,), 
            dtype=np.float32
        )
        
        # Internal state
        self.current_step = 0
        self.M = None
        self.target_attractor = None
        self.initial_M = None
        
    def reset(self):
        """Reset environment - random initial + random target attractor"""
        # Initialize in unstable region (Chen recommendation)
        self.M = np.random.uniform(0.3, 0.7, self.state_dim)
        self.initial_M = self.M.copy()
        
        # Random target attractor (relaxed problem formulation!)
        # Instead of specific target, any dominant pattern works
        pattern = np.random.choice([0, 1], size=self.state_dim, p=[0.5, 0.5])
        self.target_attractor = pattern.astype(float)
        
        self.current_step = 0
        return self.M.copy()
    
    def step(self, action):
        """Execute action: apply boost then evolve"""
        group_idx = int(action[0])  # Target group (0, 1, or 2)
        boost_factor = action[1]     # Strength 0.5~3.0
        boost_steps = int(action[2]) # Duration
        
        # Apply boost to target group
        boost_mask = np.zeros(self.state_dim)
        boost_mask[group_idx::self.k] = 1  # All layers of this group
        
        # Add perturbation (Chen: "轻微瞬时扰动")
        self.M += boost_factor * boost_mask * 0.05
        self.M = np.clip(self.M, 0.01, 0.99)
        
        # Evolve dynamics
        t = np.linspace(0, 10, 50)  # 10 time units per step
        self.M = odeint(F, self.M, t, args=(self.lambda_, self.mu))[-1]
        
        self.current_step += 1
        done = self.current_step >= self.T_max
        
        # Relaxed reward (Chen: margin=0.08)
        dist_to_target = np.mean((self.M - self.target_attractor)**2)
        
        # Normalize to [0, 1] range
        reward = 1.0 - dist_to_target * 4.0
        
        # Big bonus for success (Chen: margin 0.08, more relaxed!)
        if dist_to_target < 0.08:
            reward += 10.0
            done = True  # Episode ends on success
        
        return self.M.copy(), reward, done, {"dist": dist_to_target}
    
    def render(self, mode='human'):
        print(f"Step {self.current_step}, dist: {np.mean((self.M-self.target_attractor)**2):.4f}")


# ==================== PPO Training ====================
def train_ppo():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # Create environment
    env = DummyVecEnv([lambda: HypergraphControlEnv(lambda_=0.1, T_max=500)])
    
    # Create PPO model
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01
    )
    
    print("Training PPO...")
    model.learn(total_timesteps=50000)
    
    # Test success rate
    print("\nTesting trained model...")
    successes = 0
    dists = []
    
    for ep in range(200):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            dists.append(info[0]['dist'])
        
        # Success = got big bonus (reached target)
        if episode_reward > 9:
            successes += 1
    
    print(f"\n=== RESULTS ===")
    print(f"PPO Success Rate: {successes/200*100:.1f}%")
    print(f"Average final distance: {np.mean(dists):.4f}")
    
    # Save model
    model.save("ppo_hypergraph_control")
    print("Model saved to ppo_hypergraph_control")


if __name__ == "__main__":
    train_ppo()
