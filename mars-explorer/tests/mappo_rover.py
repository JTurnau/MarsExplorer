# MAPPO version of CleanRL PPO, adapted from ppo_atari.py

import os
import random
import time
from dataclasses import dataclass

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# Import your MarsExplorer env
from mars_explorer.envs.explorer import ExplorerMA
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    debug: bool = False

    # Algorithm specific arguments
    total_timesteps: int = 1000000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    num_steps: int = 1000
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # To be filled at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# -----------------------------
# 1️⃣ AGENT CLASS (MAPPO)
# -----------------------------
class MAPPOAgent(nn.Module):
    """
    Centralized critic, decentralized actors.
    - Actor sees global state (explored region shared map + all agent positions)
    - Critic sees global state (explored region shared map + all agent positions)
    - Parameter sharing across agents for simplicity
    """

    def __init__(self, obs_shape, action_space, n_agents):
        super().__init__()
        self.n_agents = n_agents

        obs_channels = obs_shape[2]  # H, W, C -> channels-last from environment

        # Encoder for each agent (shared across agents)
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(obs_channels, 32, kernel_size=5, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64*4*4, 512)),
            nn.ReLU()
        )

        # Actor head (per agent)
        self.actor = layer_init(nn.Linear(512, action_space[0].n), std=0.01)

        # Centralized critic: takes all agent encodings concatenated
        self.critic = layer_init(nn.Linear(512 * n_agents, 1), std=1)

    # Actor forward
    def get_actor_output(self, x):
        # x: [C,H,W]
        if x.ndim == 3:
            x = x.unsqueeze(0)  # [1, C, H, W]
        encoded = self.encoder(x)  # [1, features]
        encoded = encoded.squeeze(0)  # remove batch dimension for downstream
        logits = self.actor(encoded)
        return logits, encoded


    # Critic forward
    def get_value(self, encoded_all):
        # encoded_all: concatenated encodings of all agents [batch, 128*n_agents]
        return self.critic(encoded_all)

    # Combined step
    def get_action_and_value(self, obs, actions=None):
        """
        Supports batch input:
            obs: [batch_size, n_agents, C, H, W] or [n_agents, C, H, W]
        Returns:
            actions_out: [batch_size, n_agents] (or [n_agents] if batch_size=1)
            logprobs_out: [batch_size * n_agents]
            entropy_out: [batch_size * n_agents]
            values: [batch_size]
        """
        # Ensure obs has batch dimension
        if obs.ndim == 4:  # [n_agents, C, H, W]
            obs = obs.unsqueeze(0)  # [1, n_agents, C, H, W]

        batch_size = obs.shape[0]

        logits_list = []
        encodings_list = []

        # Loop over agents
        for agent_idx in range(self.n_agents):
            obs_agent = obs[:, agent_idx]  # [batch_size, C, H, W]
            # Pass through encoder
            encoded = self.encoder(obs_agent)  # [batch_size, features]
            logits = self.actor(encoded)       # [batch_size, action_dim]
            logits_list.append(logits)
            encodings_list.append(encoded)

        # Centralized critic sees all agent encodings
        encodings_all = torch.cat(encodings_list, dim=-1)  # [batch_size, features*n_agents]
        values = self.critic(encodings_all).squeeze(-1)    # [batch_size]

        actions_out = []
        logprobs_out = []
        entropy_out = []

        for agent_idx, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)
            if actions is None:
                action = dist.sample()
            else:
                action = actions[agent_idx]
            actions_out.append(action)
            logprobs_out.append(dist.log_prob(action))
            entropy_out.append(dist.entropy())

        # Stack along batch * agents dimension
        actions_out = torch.stack(actions_out, dim=1)     # [batch_size, n_agents]
        logprobs_out = torch.stack(logprobs_out, dim=1).reshape(-1)  # [batch_size*n_agents]
        entropy_out = torch.stack(entropy_out, dim=1).reshape(-1)    # [batch_size*n_agents]

        return actions_out, logprobs_out, entropy_out, values


# -----------------------------
# 2️⃣ MAIN TRAINING LOOP
# -----------------------------
if __name__ == "__main__":
    start_time = time.time()

    args = tyro.cli(Args)

    run_name = f"MAPPO_MarsExplorer__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    n_agents = 2  # Example for MarsExplorer
    conf["n_agents"] = n_agents
    conf["shared_map"] = True
    conf["size"] = [30, 30]
    conf["obstacles"] = 20
    conf["lidar_range"] = 4
    conf["obstacle_size"] = [1, 3]
    conf["env_mode"] = "sim"
    conf["slip_prob"] = 0.5

    seed = 42

    # create env
    env = ExplorerMA(conf=conf)

    observations = env.reset(seed=seed)

    obs_shape = env._get_obs(0).shape
    action_space = env.action_space

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    agent = MAPPOAgent(obs_shape, action_space, n_agents).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    num_steps = args.num_steps

    # increment steps
    env_step = 0
    train_step = 0

    

    num_updates = args.total_timesteps // (args.num_steps * n_agents)

    for update in range(num_updates):
        # -----------------------------
        # Storage
        # -----------------------------
        obs_tensor = torch.zeros(
            (num_steps, n_agents, obs_shape[2], obs_shape[0], obs_shape[1]),
            device=device,
        )
        actions_tensor = torch.zeros((num_steps, n_agents), device=device).long()
        logprobs_tensor = torch.zeros((num_steps, n_agents), device=device)
        rewards_tensor = torch.zeros(num_steps, device=device)   # TEAM reward
        dones_tensor = torch.zeros(num_steps, device=device)     # TEAM done
        values_tensor = torch.zeros((num_steps, 1), device=device)

        # Initial observation
        next_obs = torch.tensor(
            np.array([env._get_obs(i) for i in range(n_agents)]),
            dtype=torch.float32,
        ).permute(0, 3, 1, 2).to(device)

        next_done = 0.0  # TEAM done flag (scalar)

        # -----------------------------
        # Rollout
        # -----------------------------
        for step in range(num_steps):
            obs_tensor[step] = next_obs
            dones_tensor[step] = next_done

            with torch.no_grad():
                actions, logprobs, entropy, values = agent.get_action_and_value(next_obs)

                actions_tensor[step] = actions
                logprobs_tensor[step] = logprobs
                values_tensor[step] = values

            # Environment expects ints? ADD GAT LOGIC HERE LATER
            modified_actions = actions[0].tolist()

            next_obs_list, rewards_list, dones_list, info = env.step(modified_actions)

            # ----- TEAM reward -----
            team_reward = sum(rewards_list)
            rewards_tensor[step] = team_reward

            # ----- TEAM done -----
            episode_done = any(dones_list)
            next_done = float(episode_done)

            # ----- Debug -----
            if args.debug:
                print("\n" + "=" * 60)
                print(f"Step {step}")
                print(f"Actions: {modified_actions}")
                print(f"Rewards: {rewards_list} | Team: {team_reward}")
                print(f"Dones:   {dones_list} | Team done: {episode_done}")
                print(f"Value:   {values.item():.3f}")
                env.render()

            # ----- Reset if done -----
            if episode_done:
                if args.debug:
                    print(">>> RESETTING ENVIRONMENT <<<")

                env_step += 1
                writer.add_scalar("charts/episodic_return", team_reward, env_step)
                writer.add_scalar("charts/episodic_length", step + 1, env_step)

                next_obs_list = env.reset(seed=seed)
                next_done = 0.0

            # Convert obs
            next_obs = torch.tensor(
                np.array(next_obs_list),
                dtype=torch.float32,
            ).permute(0, 3, 1, 2).to(device)

            if args.debug:
                input("Press ENTER to continue...")

        # -----------------------------
        # GAE (CENTRALIZED CRITIC)
        # -----------------------------
        with torch.no_grad():
            advantages = torch.zeros(num_steps, device=device)
            lastgaelam = 0.0

            # Bootstrap last value
            last_value = agent.get_action_and_value(next_obs)[-1]  # last element is values

            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_value = last_value
                else:
                    next_nonterminal = 1.0 - dones_tensor[t + 1]
                    next_value = values_tensor[t + 1]

                delta = (
                    rewards_tensor[t]
                    + args.gamma * next_value * next_nonterminal
                    - values_tensor[t]
                )

                advantages[t] = lastgaelam = (
                    delta
                    + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
                )

            returns = advantages + values_tensor.squeeze(-1)

        # -----------------------------
        # Flatten rollout for training
        # -----------------------------
        b_obs = obs_tensor.permute(0, 1, 3, 4, 2).reshape(-1, obs_shape[2], obs_shape[0], obs_shape[1]).to(device)
        b_actions = actions_tensor.reshape(-1).to(device)
        b_logprobs = logprobs_tensor.reshape(-1).to(device)

        # Repeat returns, advantages, and values across agents
        b_returns = returns.repeat(n_agents)
        b_values = values_tensor.squeeze(-1).repeat(n_agents)
        b_advantages = advantages.repeat(n_agents)

        # Move to device
        b_returns = b_returns.to(device)
        b_values = b_values.to(device)
        b_advantages = b_advantages.to(device)



        # normalize advantages
        if args.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)


        batch_size = b_obs.shape[0]
        minibatch_size = batch_size // args.num_minibatches

        for epoch in range(args.update_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                mb_obs = b_obs[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_logprobs = b_logprobs[mb_idx]
                mb_returns = b_returns[mb_idx]
                mb_advantages = b_advantages[mb_idx]

                # Forward pass
                # Reshape mb_actions into a list of tensors, one per agent
                mb_actions_list = [
                    mb_actions[i::n_agents] for i in range(n_agents)
                ]  # each tensor is shape [batch_size_per_agent]

                new_actions, new_logprobs, new_entropy, new_values = agent.get_action_and_value(
                    mb_obs, actions=mb_actions_list
                )

                # new_logprobs and new_entropy are lists per agent, flatten them
                new_logprobs = torch.cat([lp.reshape(-1) for lp in new_logprobs])
                new_entropy = torch.cat([e.reshape(-1) for e in new_entropy])

                new_values = new_values.squeeze(-1)

                # PPO policy loss
                ratio = (new_logprobs - mb_logprobs).exp()
                policy_loss = -(torch.min(
                    ratio * mb_advantages,
                    torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * mb_advantages
                )).mean()

                # Value loss
                if args.clip_vloss:
                    v_loss_unclipped = (new_values - mb_returns).pow(2)
                    v_clipped = b_values[mb_idx] + (new_values - b_values[mb_idx]).clamp(-args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - mb_returns).pow(2)
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = 0.5 * (new_values - mb_returns).pow(2).mean()

                # Entropy loss
                entropy_loss = new_entropy.mean()

                # Total loss
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_loss

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # Logging
                train_step += 1  # increment training step
                writer.add_scalar("losses/value_loss", value_loss.item(), train_step)
                writer.add_scalar("losses/policy_loss", policy_loss.item(), train_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), train_step)
        
        # --- End of update ---
        print(f"Update {update+1}/{num_updates} done")


end_time = time.time()
elapsed = end_time - start_time

print("=" * 60)
print(f"Total training time: {elapsed / 60:.2f} minutes")
print(f"Total training time: {elapsed / 3600:.2f} hours")
print("=" * 60)