# -----------------------------
# IPPO for 2-agent Mars Rover (Multi-Seed Training)
# -----------------------------
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from mars_explorer.envs.explorer import ExplorerMA
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf

# -----------------------------
# Helper functions
# -----------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

# -----------------------------
# PPO Agent (Independent) - Improved Architecture
# -----------------------------
class IPPOAgent(nn.Module):
    def __init__(self, obs_shape, action_space):
        super().__init__()
        obs_channels = obs_shape[2]
        H, W = obs_shape[0], obs_shape[1]

        # Improved CNN architecture with more capacity
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(obs_channels, 32, 3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=2, padding=1)),  # 30x30 -> 15x15
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, obs_channels, H, W)
            dummy_output = self.encoder(dummy_input)
            conv_out_size = dummy_output.shape[1]
        
        # Shared feature layer
        self.fc = nn.Sequential(
            layer_init(nn.Linear(conv_out_size, 256)),
            nn.ReLU(),
        )

        # Actor head
        self.actor = layer_init(nn.Linear(256, action_space.n), std=0.01)
        # Critic head
        self.critic = layer_init(nn.Linear(256, 1), std=1.0)

    def get_value(self, obs):
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        features = self.encoder(obs)
        features = self.fc(features)
        value = self.critic(features).squeeze(-1)
        return value

    def get_action_and_value(self, obs, actions=None):
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        features = self.encoder(obs)
        features = self.fc(features)
        logits = self.actor(features)
        dist = Categorical(logits=logits)
        action = dist.sample() if actions is None else actions
        logprob = dist.log_prob(action)
        value = self.critic(features).squeeze(-1)
        entropy = dist.entropy()
        return action, logprob, entropy, value

    def get_deterministic_action(self, obs):
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        features = self.encoder(obs)
        features = self.fc(features)
        logits = self.actor(features)
        action = torch.argmax(logits, dim=-1)
        return action

# -----------------------------
# Training parameters
# -----------------------------
n_agents = 2
conf["n_agents"] = n_agents
conf["shared_map"] = True
conf["size"] = [30, 30]
conf["obstacles"] = 20
conf["lidar_range"] = 4
conf["obstacle_size"] = [1, 3]
conf["env_mode"] = "sim"
conf["slip_prob"] = 0.0
conf["initial"] = [1, 1]
conf["collision_reward"] = -5
conf["out_of_bounds_reward"] = -5
conf["movementCost"] = 0.1  # Keep original

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
num_steps = 256  # Increased for more data per update
num_updates = 500
num_minibatches = 4
update_epochs = 4
gamma, gae_lambda = 0.99, 0.95
clip_coef = 0.2
ent_coef = 0.1  # Higher entropy for more exploration
vf_coef = 0.5
max_grad_norm = 0.5
lr = 3e-4
anneal_lr = True

# Multi-seed configuration
num_train_seeds = 10  # Train on 10 different seeds
train_seed_base = 42  # Starting seed
num_test_seeds = 5    # Test on 5 different seeds
test_seed_base = 1000  # Starting test seed

save_dir = "checkpoints_ippo"
os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# Initialize environment and agents
# -----------------------------
env = ExplorerMA(conf)
obs_list = env.reset(seed=train_seed_base)
obs_shape = np.array(obs_list[0]).shape

agents = [IPPOAgent(obs_shape, env.action_space[0]).to(device) for _ in range(n_agents)]
optimizers = [optim.Adam(agent.parameters(), lr=lr, eps=1e-5) for agent in agents]

writer = SummaryWriter("runs/IPPO_multiseed")

best_return = -np.inf

# -----------------------------
# Training Loop
# -----------------------------
def train_agents():
    global best_return
    global_step = 0
    
    for update in range(1, num_updates + 1):
        # Anneal learning rate
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * lr
            for optimizer in optimizers:
                optimizer.param_groups[0]["lr"] = lrnow
        
        # Anneal entropy coefficient (start high, gradually decrease)
        frac = 1.0 - (update - 1.0) / num_updates
        ent_coef_now = ent_coef * max(0.3, frac)  # Don't go below 30% of initial
        
        # Rollout storage per agent
        obs_buffer = [torch.zeros(num_steps, obs_shape[2], obs_shape[0], obs_shape[1], 
                                   device=device, dtype=torch.float32) for _ in range(n_agents)]
        actions_buffer = [torch.zeros(num_steps, device=device).long() for _ in range(n_agents)]
        logprobs_buffer = [torch.zeros(num_steps, device=device) for _ in range(n_agents)]
        values_buffer = [torch.zeros(num_steps, device=device) for _ in range(n_agents)]
        rewards_buffer = [torch.zeros(num_steps, device=device) for _ in range(n_agents)]
        dones_buffer = [torch.zeros(num_steps, device=device) for _ in range(n_agents)]
        
        # Episode tracking
        episode_returns = [[] for _ in range(n_agents)]
        episode_lengths = [[] for _ in range(n_agents)]
        current_ep_returns = [0.0] * n_agents
        current_ep_lengths = [0] * n_agents

        # DIVERSE SEED SELECTION: Cycle through different seeds each update
        current_seed = train_seed_base + (update % num_train_seeds)
        obs_list = env.reset(seed=current_seed)
        next_obs_list = [torch.tensor(o, dtype=torch.float32, device=device).permute(2,0,1) for o in obs_list]

        for step in range(num_steps):
            global_step += 1
            
            # Get actions for all agents
            actions_list, logprobs_list, entropy_list, values_list = [], [], [], []
            for agent_idx in range(n_agents):
                with torch.no_grad():
                    a, lp, ent, val = agents[agent_idx].get_action_and_value(next_obs_list[agent_idx])
                actions_list.append(a)
                logprobs_list.append(lp)
                entropy_list.append(ent)
                values_list.append(val)

            # Store data
            for agent_idx in range(n_agents):
                obs_buffer[agent_idx][step] = next_obs_list[agent_idx]
                actions_buffer[agent_idx][step] = actions_list[agent_idx]
                logprobs_buffer[agent_idx][step] = logprobs_list[agent_idx]
                values_buffer[agent_idx][step] = values_list[agent_idx]

            # Step environment
            env_actions = [a.item() for a in actions_list]
            next_obs_raw, rewards, dones, info = env.step(env_actions)

            # Store rewards and dones
            for agent_idx in range(n_agents):
                rewards_buffer[agent_idx][step] = rewards[agent_idx]
                dones_buffer[agent_idx][step] = float(dones[agent_idx])
                
                current_ep_returns[agent_idx] += rewards[agent_idx]
                current_ep_lengths[agent_idx] += 1

            next_obs_list = [torch.tensor(o, dtype=torch.float32, device=device).permute(2,0,1) for o in next_obs_raw]

            # Handle episode termination
            if any(dones):
                for agent_idx in range(n_agents):
                    if dones[agent_idx]:
                        episode_returns[agent_idx].append(current_ep_returns[agent_idx])
                        episode_lengths[agent_idx].append(current_ep_lengths[agent_idx])
                        current_ep_returns[agent_idx] = 0.0
                        current_ep_lengths[agent_idx] = 0
                
                # Reset with SAME seed to maintain consistency within rollout
                obs_list = env.reset(seed=current_seed)
                next_obs_list = [torch.tensor(o, dtype=torch.float32, device=device).permute(2,0,1) for o in obs_list]

        # Bootstrap value if episode didn't end
        with torch.no_grad():
            next_values = [agents[i].get_value(next_obs_list[i]) for i in range(n_agents)]

        # Compute advantages and returns using GAE
        advantages_list = []
        returns_list = []
        
        for agent_idx in range(n_agents):
            advantages = torch.zeros_like(rewards_buffer[agent_idx])
            lastgaelam = 0
            
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - dones_buffer[agent_idx][t]
                    nextvalues = next_values[agent_idx]
                else:
                    nextnonterminal = 1.0 - dones_buffer[agent_idx][t]
                    nextvalues = values_buffer[agent_idx][t + 1]
                
                delta = rewards_buffer[agent_idx][t] + gamma * nextvalues * nextnonterminal - values_buffer[agent_idx][t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values_buffer[agent_idx]
            advantages_list.append(advantages)
            returns_list.append(returns)

        # PPO update for each agent
        clipfracs = []
        for agent_idx in range(n_agents):
            # Flatten the batch
            b_obs = obs_buffer[agent_idx].reshape((-1, obs_shape[2], obs_shape[0], obs_shape[1]))
            b_actions = actions_buffer[agent_idx].reshape(-1)
            b_logprobs = logprobs_buffer[agent_idx].reshape(-1)
            b_advantages = advantages_list[agent_idx].reshape(-1)
            b_returns = returns_list[agent_idx].reshape(-1)
            b_values = values_buffer[agent_idx].reshape(-1)

            # Normalize advantages
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # Optimizing the policy and value network
            batch_size = num_steps
            minibatch_size = batch_size // num_minibatches
            
            for epoch in range(update_epochs):
                # Generate random indices for minibatches
                b_inds = np.arange(batch_size)
                np.random.shuffle(b_inds)
                
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agents[agent_idx].get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # Calculate approx_kl for monitoring
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
                        clipfracs.append(clipfrac.item())

                    mb_advantages = b_advantages[mb_inds]

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - ent_coef_now * entropy_loss + v_loss * vf_coef

                    optimizers[agent_idx].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agents[agent_idx].parameters(), max_grad_norm)
                    optimizers[agent_idx].step()

            # Logging per agent
            if len(episode_returns[agent_idx]) > 0:
                writer.add_scalar(f"agent{agent_idx}/episodic_return", np.mean(episode_returns[agent_idx]), global_step)
                writer.add_scalar(f"agent{agent_idx}/episodic_length", np.mean(episode_lengths[agent_idx]), global_step)
            
            writer.add_scalar(f"agent{agent_idx}/value_loss", v_loss.item(), global_step)
            writer.add_scalar(f"agent{agent_idx}/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar(f"agent{agent_idx}/entropy", entropy_loss.item(), global_step)

        # Global logging
        writer.add_scalar("charts/learning_rate", optimizers[0].param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/entropy_coef", ent_coef_now, global_step)
        writer.add_scalar("charts/clipfrac", np.mean(clipfracs), global_step)

        # Save best model based on average return across agents
        avg_returns = []
        for agent_idx in range(n_agents):
            if len(episode_returns[agent_idx]) > 0:
                avg_returns.append(np.mean(episode_returns[agent_idx]))
        
        if len(avg_returns) > 0:
            avg_return = np.mean(avg_returns)
            writer.add_scalar("charts/avg_return", avg_return, global_step)
            
            if avg_return > best_return:
                best_return = avg_return
                for agent_idx, agent in enumerate(agents):
                    torch.save(agent.state_dict(), os.path.join(save_dir, f"best_agent{agent_idx+1}.pt"))
                print(f"[Update {update}] New best avg return: {avg_return:.2f} (seed {current_seed}) | Models saved.")
            
            print(f"Update {update}/{num_updates} | Seed: {current_seed} | AvgReturn: {avg_return:.2f} | Ent: {ent_coef_now:.3f}")
        else:
            print(f"Update {update}/{num_updates} | Seed: {current_seed} | No episodes completed")

# -----------------------------
# Testing / Evaluation
# -----------------------------
def test_agents(num_episodes_per_seed=1):
    """Test on multiple seeds to get robust performance estimate"""
    test_env = ExplorerMA(conf)
    agents_loaded = []
    for agent_idx in range(n_agents):
        agent = IPPOAgent(obs_shape, test_env.action_space[0]).to(device)
        agent.load_state_dict(torch.load(os.path.join(save_dir, f"best_agent{agent_idx+1}.pt")))
        agent.eval()
        agents_loaded.append(agent)

    all_returns = []
    all_lengths = []
    
    for seed_idx in range(num_test_seeds):
        test_env = ExplorerMA(conf)
        test_seed = test_seed_base + seed_idx
        print(f"\n{'='*60}")
        print(f"Testing on seed {test_seed}")
        print(f"{'='*60}")
        
        seed_returns = []
        seed_lengths = []
        
        for ep in range(num_episodes_per_seed):
            obs_list = test_env.reset(seed=test_seed)
            done = [False] * n_agents
            total_rewards = [0.0] * n_agents
            test_steps = 0

            # Render first episode of each seed
            if ep == 0:
                test_env.render()
                time.sleep(0.1)

            while not all(done) and test_steps < 30:
                actions_list = []
                for agent_idx in range(n_agents):
                    obs = torch.tensor(obs_list[agent_idx], dtype=torch.float32, device=device).permute(2, 0, 1)
                    with torch.no_grad():
                        action = agents_loaded[agent_idx].get_deterministic_action(obs)
                    actions_list.append(action.item())

                obs_list, rewards, done, info = test_env.step(actions_list)
                total_rewards = [tr + r for tr, r in zip(total_rewards, rewards)]

                if ep == 0:  # Only print for first episode
                    print(f"Step {test_steps}: actions: {actions_list}, rewards: {rewards}")
                    test = input("...")

                if ep == 0:
                    test_env.render()
                    time.sleep(0.1)

                test_steps += 1

            ep_avg_return = np.mean(total_rewards)
            seed_returns.append(ep_avg_return)
            seed_lengths.append(test_steps)
            
            print(f"\nEpisode {ep+1}/{num_episodes_per_seed}: Steps={test_steps}, Returns={total_rewards}, Avg={ep_avg_return:.2f}")
        
        # Statistics for this seed
        mean_return = np.mean(seed_returns)
        std_return = np.std(seed_returns)
        mean_length = np.mean(seed_lengths)
        
        print(f"\nSeed {test_seed} Summary:")
        print(f"  Mean Return: {mean_return:.2f} ± {std_return:.2f}")
        print(f"  Mean Length: {mean_length:.1f}")
        
        all_returns.extend(seed_returns)
        all_lengths.extend(seed_lengths)
    
    # Overall statistics across all test seeds
    print(f"\n{'='*60}")
    print(f"OVERALL TEST PERFORMANCE")
    print(f"{'='*60}")
    print(f"Total Episodes: {len(all_returns)} ({num_test_seeds} seeds × {num_episodes_per_seed} episodes)")
    print(f"Mean Return: {np.mean(all_returns):.2f} ± {np.std(all_returns):.2f}")
    print(f"Mean Length: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")
    print(f"Min Return: {np.min(all_returns):.2f}")
    print(f"Max Return: {np.max(all_returns):.2f}")
    print(f"{'='*60}\n")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    TRAIN = False

    if TRAIN:
        print(f"Training Configuration:")
        print(f"  Training seeds: {train_seed_base} to {train_seed_base + num_train_seeds - 1}")
        print(f"  Test seeds: {test_seed_base} to {test_seed_base + num_test_seeds - 1}")
        print(f"  Updates: {num_updates}")
        print(f"  Rollout length: {num_steps}")
        print(f"  Initial entropy coef: {ent_coef}")
        print(f"  Learning rate: {lr}")
        print()
        
        train_agents()
        test_agents()
    else:
        test_agents()