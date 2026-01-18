"""
TD3 (Twin Delayed DDPG) Agent for ROS2 RL Training
Adapted from rl_model_based/agents/td3_gazebo.py

Implements TD3 algorithm with:
- Actor network (policy)
- Twin critic networks (Q1, Q2)
- Target networks with soft updates
- Delayed policy updates
- Target policy smoothing
"""

import os
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ============================================================================
# TD3 HYPERPARAMETERS & CONFIGURATION
# ============================================================================

# Learning rates
TD3_ACTOR_LR = 3e-4
TD3_CRITIC_LR = 3e-4

# Discount and update rates
TD3_GAMMA = 0.99
TD3_TAU = 0.005

# TD3-specific: Policy smoothing and delayed updates
TD3_POLICY_NOISE = 0.2      # Noise added to target policy during critic update
TD3_NOISE_CLIP = 0.5        # Range to clip target policy noise
TD3_POLICY_DELAY = 2        # Delay between policy updates (update every N iterations)

# Replay buffer and batch size
TD3_BUFFER_SIZE = int(1e6)
TD3_BATCH_SIZE = 256

def _to_tensor(x, device):
    """Convert numpy array to torch tensor"""
    return torch.tensor(x, dtype=torch.float32, device=device)


# ============================================================================
# NEURAL NETWORKS
# ============================================================================

class Actor(nn.Module):
    """
    Actor network: maps state → action
    
    Architecture: state_dim → 400 → 300 → action_dim
    Output: tanh activation scaled by max_action
    """
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x * self.max_action


class Critic(nn.Module):
    """
    Twin Critic networks: map (state, action) → Q-value
    
    Uses two Q-networks (Q1, Q2) to reduce overestimation bias.
    Architecture: (state_dim + action_dim) → 400 → 300 → 1
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1 network
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        
        # Q2 network
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)
    
    def forward(self, state, action):
        """Return both Q1 and Q2 values"""
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        """Return only Q1 value (for actor loss)"""
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning
    
    Stores transitions (s, a, r, s', done) and samples random batches.
    """
    def __init__(self, max_size=int(1e6)):
        self.storage = deque(maxlen=int(max_size))
    
    def add(self, state, action, next_state, reward, done):
        """Add transition (s, a, s', r, done)"""
        self.storage.append((state, action, next_state, reward, done))
    
    def sample(self, batch_size):
        """Sample random batch"""
        batch = random.sample(self.storage, batch_size)
        state, action, next_state, reward, done = map(np.stack, zip(*batch))
        return (
            state,
            action,
            next_state,
            reward.reshape(-1, 1),
            done.reshape(-1, 1)
        )
    
    def size(self):
        """Current buffer size"""
        return len(self.storage)
    
    def save(self, filepath):
        """Save buffer to disk"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.storage), f)
        print(f"Replay buffer saved to {filepath}")
    
    def load(self, filepath):
        """Load buffer from disk"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.storage = deque(data, maxlen=self.storage.maxlen)
        print(f"Replay buffer loaded from {filepath} ({len(self.storage)} transitions)")


# ============================================================================
# TD3 AGENT
# ============================================================================

class TD3Agent:
    """
    TD3 (Twin Delayed DDPG) Agent
    
    Key features:
    - Twin Q-networks to reduce overestimation
    - Delayed policy updates (update actor less frequently)
    - Target policy smoothing (add noise to target actions)
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action=1.0,
        min_action=-1.0,
        actor_lr=TD3_ACTOR_LR,
        critic_lr=TD3_CRITIC_LR,
        gamma=TD3_GAMMA,
        tau=TD3_TAU,
        policy_noise=TD3_POLICY_NOISE,
        noise_clip=TD3_NOISE_CLIP,
        policy_delay=TD3_POLICY_DELAY,
        buffer_size=TD3_BUFFER_SIZE,
        batch_size=TD3_BATCH_SIZE,
        device=None,
        seed=0,
    ):
        """
        Initialize TD3 agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_action: Maximum action value
            min_action: Minimum action value
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            policy_noise: Noise added to target policy
            noise_clip: Clip range for policy noise
            policy_delay: Delay between policy updates
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            device: torch device (cuda/cpu)
            seed: Random seed
        """
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Handle array inputs for max_action and min_action
        self.max_action = float(np.max(max_action)) if isinstance(max_action, (list, np.ndarray)) else float(max_action)
        self.min_action = float(np.min(min_action)) if isinstance(min_action, (list, np.ndarray)) else float(min_action)
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        
        # Create actor and target actor
        self.actor = Actor(state_dim, action_dim, self.max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Create critic and target critic
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        
        # Training counter
        self.total_it = 0
        
        # Checkpoint directory
        self.checkpoint_dir = 'checkpoints/td3'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"TD3 Agent initialized:")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  Device: {self.device}")
        print(f"  Buffer size: {buffer_size}, Batch size: {batch_size}")
    
    def select_action(self, state, evaluate=False):
        """
        Select action using current policy
        
        Args:
            state: Current state
            evaluate: If True, no exploration noise
        
        Returns:
            Action array
        """
        if isinstance(state, np.ndarray):
            state = state.reshape(1, -1)
        state_t = _to_tensor(state, self.device)
        
        # Save training mode
        was_training = self.actor.training
        
        # Inference mode
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t).cpu().data.numpy().flatten()
        
        # Restore training mode
        if was_training:
            self.actor.train()
        
        # Add exploration noise
        if not evaluate:
            noise = np.random.normal(0, 0.1, size=self.action_dim)
            action = np.clip(action + noise, self.min_action, self.max_action)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        reward = float(reward)
        done = float(done)
        self.replay_buffer.add(state, action, next_state, reward, done)
    
    def train(self):
        """
        Train agent using TD3 algorithm
        
        Returns:
            Tuple of (actor_loss, critic_loss) or (None, None) if not enough samples
        """
        if self.replay_buffer.size() < max(self.batch_size, 1000):
            return None, None
        
        # Ensure networks are in training mode for gradient updates
        self.actor.train()
        self.critic.train()
        
        self.total_it += 1
        
        # Sample batch
        state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        not_done = torch.FloatTensor(not_done).to(self.device)
        
        with torch.no_grad():
            # Target policy smoothing: add clipped noise to target actions
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                self.min_action, self.max_action
            )
            
            # Compute target Q-value (minimum of Q1 and Q2)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (not_done * self.gamma * target_Q)
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = None
        
        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Actor loss: maximize Q1(s, actor(s))
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
        
        return (
            actor_loss.item() if actor_loss is not None else None,
            critic_loss.item()
        )
    
    def save(self, filename='td3_best'):
        """Save actor and critic networks"""
        actor_path = os.path.join(self.checkpoint_dir, f'actor_{filename}.pth')
        critic_path = os.path.join(self.checkpoint_dir, f'critic_{filename}.pth')
        
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        
        print(f"Models saved: {filename}")
    
    def save_models(self, episode=None):
        """Alias for save() to match SAC API"""
        if episode is None:
            self.save('td3_best')
        else:
            self.save(f'td3_ep_{episode}')
    
    def load(self, filename='td3_best'):
        """Load actor and critic networks"""
        actor_path = os.path.join(self.checkpoint_dir, f'actor_{filename}.pth')
        critic_path = os.path.join(self.checkpoint_dir, f'critic_{filename}.pth')
        
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        
        # Copy to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        print(f"Models loaded: {filename}")
    
    def load_models(self, actor_path, critic_path=None):
        """
        Load actor and critic networks from paths.
        Matches SAC API for consistent usage in train_robot.py
        
        Args:
            actor_path: Path to actor checkpoint
            critic_path: Path to critic checkpoint (optional, will infer if not provided)
        """
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        
        # Infer critic path from actor path if not provided
        if critic_path is None:
            critic_path = actor_path.replace('actor_', 'critic_')
        
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        
        # Copy to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Set to eval mode
        self.actor.eval()
        self.critic.eval()
        
        print(f"✅ TD3 models loaded from: {actor_path}")
