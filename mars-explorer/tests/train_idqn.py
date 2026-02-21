import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
from datetime import datetime

from mars_explorer.envs.explorer import ExplorerMALocalObs
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf


class DQN_CNN(nn.Module):
    """
    CNN-based DQN for processing local observations (supports any square size).
    Single channel input representing the local grid around the agent.
    """
    def __init__(self, obs_size=7, n_actions=4):
        super(DQN_CNN, self).__init__()
        
        self.obs_size = obs_size
        
        # Input: (batch, 1, obs_size, obs_size)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate flattened size after convolutions
        # With padding=1 and stride=1, spatial dimensions stay the same
        self.flattened_size = 64 * obs_size * obs_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_actions)
        
    def forward(self, x):
        # x shape: (batch, obs_size, obs_size) -> add channel dimension
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # (batch, 1, obs_size, obs_size)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class ReplayBuffer:
    """
    Experience replay buffer for storing transitions from both agents.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class IndependentDQN:
    """
    Independent DQN with parameter sharing.
    Single model is trained on experiences from all agents.
    """
    def __init__(self, obs_size=7, n_actions=4, learning_rate=1e-4, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, 
                 epsilon_decay=0.995, buffer_capacity=100000, 
                 batch_size=128, target_update_freq=1000, device='cuda'):
        
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        
        # Single policy network (shared by all agents)
        self.policy_net = DQN_CNN(obs_size, n_actions).to(device)
        self.target_net = DQN_CNN(obs_size, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        self.steps_done = 0
        self.update_counter = 0
        
    def select_action(self, state, eval_mode=False):
        """
        Select action using epsilon-greedy policy.
        State: (obs_size, obs_size) numpy array
        """
        if eval_mode or random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.n_actions)
        
        self.steps_done += 1
        return action
    
    def update_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'update_counter': self.update_counter
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.update_counter = checkpoint['update_counter']
        print(f"Model loaded from {path}")


def evaluate_deterministic(env_config, agent, obs_size, n_eval_episodes=1, seed=22):
    """
    Run deterministic evaluation episodes.
    
    Args:
        env_config: Environment configuration dictionary
        agent: Trained agent
        obs_size: Size of local observation window
        n_eval_episodes: Number of evaluation episodes
        seed: Seed for reproducibility
    
    Returns:
        avg_reward: Average reward across evaluation episodes
        avg_coverage: Average coverage across evaluation episodes
        avg_length: Average episode length
    """
    env = ExplorerMALocalObs(conf=env_config)
    n_agents = env.n_agents
    
    eval_rewards = []
    eval_coverages = []
    eval_lengths = []
    
    for _ in range(n_eval_episodes):
        obs, info = env.reset(seed=seed)
        episode_reward = [0] * n_agents
        episode_length = 0
        done = False
        
        while not done:
            # Select actions deterministically (no exploration)
            actions = [agent.select_action(obs[i], eval_mode=True) for i in range(n_agents)]
            
            # Step environment
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            for i in range(n_agents):
                episode_reward[i] += rewards[i]
            
            obs = next_obs
            episode_length += 1
        
        # Calculate coverage
        total_cells = env.SIZE[0] * env.SIZE[1]
        explored_cells = np.count_nonzero(env.exploredMap)
        coverage = explored_cells / total_cells
        
        eval_rewards.append(np.mean(episode_reward))
        eval_coverages.append(coverage)
        eval_lengths.append(episode_length)
    
    return np.mean(eval_rewards), np.mean(eval_coverages), np.mean(eval_lengths)


def train_idqn(env_config, n_episodes=5000, save_freq=100, log_freq=10, 
               checkpoint_dir='checkpoints', device='cuda', obs_size=7, eval_freq=10):
    """
    Train Independent DQN with parameter sharing on multi-agent environment.
    
    Args:
        env_config: Environment configuration dictionary
        n_episodes: Number of training episodes
        save_freq: Save checkpoint every N episodes
        log_freq: Log progress every N episodes
        checkpoint_dir: Directory to save checkpoints
        device: 'cuda' or 'cpu'
        obs_size: Size of local observation window (7 for 7x7)
        eval_freq: Run deterministic evaluation every N episodes
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(checkpoint_dir, f'idqn_run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Initialize environment
    env = ExplorerMALocalObs(conf=env_config)
    n_agents = env.n_agents
    
    # Initialize agent with correct observation size
    agent = IndependentDQN(
        obs_size=obs_size,
        n_actions=4,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_capacity=100000,
        batch_size=128,
        target_update_freq=1000,
        device=device
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []
    coverage_rates = []
    
    # Evaluation metrics
    eval_rewards = []
    eval_coverages = []
    eval_lengths = []
    eval_episodes = []
    
    # Track best model
    best_eval_reward = -float('inf')
    
    print(f"Starting training with {n_agents} agents")
    print(f"Observation size: {obs_size}x{obs_size}")
    print(f"Device: {device}")
    print(f"Checkpoint directory: {run_dir}")
    print(f"Evaluation frequency: every {eval_freq} episodes")
    print("-" * 60)
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=22)
        episode_reward = [0] * n_agents
        episode_length = 0
        done = False
        
        while not done:
            # Select actions for all agents using the shared policy
            actions = [agent.select_action(obs[i]) for i in range(n_agents)]
            
            # Step environment
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            # Store transitions for all agents
            for i in range(n_agents):
                agent.store_transition(obs[i], actions[i], rewards[i], next_obs[i], done)
                episode_reward[i] += rewards[i]
            
            # Train the network
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            
            obs = next_obs
            episode_length += 1
        
        # Update epsilon
        agent.update_epsilon()
        
        # Calculate coverage
        total_cells = env.SIZE[0] * env.SIZE[1]
        explored_cells = np.count_nonzero(env.exploredMap)
        coverage = explored_cells / total_cells
        
        # Store metrics
        avg_reward = np.mean(episode_reward)
        episode_rewards.append(avg_reward)
        episode_lengths.append(episode_length)
        coverage_rates.append(coverage)
        
        # Run deterministic evaluation
        if (episode + 1) % eval_freq == 0:
            eval_reward, eval_coverage, eval_length = evaluate_deterministic(
                env_config, agent, obs_size, n_eval_episodes=1, seed=22
            )
            eval_rewards.append(eval_reward)
            eval_coverages.append(eval_coverage)
            eval_lengths.append(eval_length)
            eval_episodes.append(episode + 1)
            
            print(f"\n{'='*60}")
            print(f"DETERMINISTIC EVALUATION at Episode {episode + 1}")
            print(f"  Eval Reward: {eval_reward:.2f}")
            print(f"  Eval Coverage: {eval_coverage:.2%}")
            print(f"  Eval Length: {eval_length:.1f}")
            print(f"{'='*60}\n")
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_model_path = os.path.join(run_dir, 'best_model.pt')
                agent.save(best_model_path)
                print(f"*** NEW BEST MODEL saved with eval reward: {eval_reward:.2f} ***\n")
        
        # Logging
        if (episode + 1) % log_freq == 0:
            avg_loss = np.mean(losses[-100:]) if losses else 0
            avg_reward_recent = np.mean(episode_rewards[-log_freq:])
            avg_length_recent = np.mean(episode_lengths[-log_freq:])
            avg_coverage_recent = np.mean(coverage_rates[-log_freq:])
            
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"  Avg Reward (last {log_freq}): {avg_reward_recent:.2f}")
            print(f"  Avg Length (last {log_freq}): {avg_length_recent:.1f}")
            print(f"  Avg Coverage (last {log_freq}): {avg_coverage_recent:.2%}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Buffer Size: {len(agent.replay_buffer)}")
            print(f"  Best Eval Reward: {best_eval_reward:.2f}")
            print("-" * 60)
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint_path = os.path.join(run_dir, f'checkpoint_ep{episode + 1}.pt')
            agent.save(checkpoint_path)
            
            # Save metrics
            metrics = {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'coverage_rates': coverage_rates,
                'losses': losses,
                'eval_rewards': eval_rewards,
                'eval_coverages': eval_coverages,
                'eval_lengths': eval_lengths,
                'eval_episodes': eval_episodes,
                'best_eval_reward': best_eval_reward
            }
            metrics_path = os.path.join(run_dir, 'metrics.npy')
            np.save(metrics_path, metrics)
    
    # Save final model
    final_path = os.path.join(run_dir, 'final_model.pt')
    agent.save(final_path)
    
    print("\nTraining completed!")
    print(f"Final model saved to {final_path}")
    print(f"Best model saved to {os.path.join(run_dir, 'best_model.pt')}")
    print(f"Best eval reward: {best_eval_reward:.2f}")
    
    return agent, episode_rewards, episode_lengths, coverage_rates


if __name__ == '__main__':
    conf["n_agents"] = 2
    conf["shared_map"] = True
    conf["size"] = [15, 15]
    conf["obstacles"] = 10
    conf["lidar_range"] = 2
    conf["obstacle_size"] = [1, 3]
    conf["env_mode"] = "sim"
    conf["slip_prob"] = 0.0
    conf["initial"] = [1, 1]
    conf["collision_reward"] = -50
    conf["out_of_bounds_reward"] = -50
    conf["movementCost"] = 0.1
    conf["max_steps"] = 200

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Train the agent with 7x7 observations
    agent, rewards, lengths, coverage = train_idqn(
        env_config=conf,
        n_episodes=1500,
        save_freq=100,
        log_freq=10,
        checkpoint_dir='checkpoints',
        device=device,
        obs_size=7,
        eval_freq=10
    )