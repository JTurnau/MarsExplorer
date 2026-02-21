"""
Multi-Agent Grounded Action Transformation (MA-GAT) for Sim-to-Real Transfer

This script implements two variants of GAT for a multi-agent exploration task:
  1. Shared GAT   – all agents share a single forward model and a single inverse model
  2. Decentralized GAT – each agent has its own forward model and inverse model

Pipeline:
  1. Collect real-world trajectories (sim-to-sim with different dynamics / slip)
  2. Collect simulation trajectories
  3. Train forward models on real trajectories (learns P_real(s'|s,a))
  4. Train inverse models on sim trajectories (learns grounded action π_inv(s,s'))
  5. Fine-tune the DQN policy in sim using the grounded (transformed) actions

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
import json
import argparse
from datetime import datetime
from copy import deepcopy

from mars_explorer.envs.explorer import ExplorerMALocalObs
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf
from train_idqn import DQN_CNN, IndependentDQN, ReplayBuffer


# ---------------------------------------------------------------------------
# GAT Neural-Network Components
# ---------------------------------------------------------------------------

class ForwardModel(nn.Module):
    """
    Predicts the next observation in the *real* environment.

    Input:  flat(obs_t)  ||  one_hot(action_t)   dim = obs_dim + n_actions
    Output: flat(obs_{t+1})                       dim = obs_dim

    Trained exclusively on real-world (high-slip) trajectories.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim   = obs_dim
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, obs: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action_onehot], dim=-1)
        return self.net(x)


class InverseModel(nn.Module):
    """
    Produces a *grounded* action that bridges sim and real dynamics.

    Input:  flat(obs_t)  ||  flat(obs_{t+1})   dim = 2 * obs_dim
    Output: logits over actions                  dim = n_actions

    Training: Trained on simulation (obs, action, next_obs) triples.
              Learns to reconstruct the action that caused a given (obs → next_obs)
              transition in sim. This means at inference time, when we feed it
              (obs, forward_model_prediction), it will output whichever sim action
              is most likely to produce that predicted next state.

    Inference: Receives (obs_t, forward_model(obs_t, intended_action)) as input,
               where the forward model has been trained on *real* dynamics.
               The inverse model therefore maps the real-predicted next state back
               to the sim action that best replicates it — the grounded action.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim   = obs_dim
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Linear(2 * obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, next_obs], dim=-1)
        return self.net(x)   # raw logits; caller applies argmax or softmax


# ---------------------------------------------------------------------------
# GAT Agent Wrappers
# ---------------------------------------------------------------------------

class SharedGAT:
    """
    Single forward + inverse model shared across all agents (parameter sharing).
    """

    def __init__(self, obs_size: int, n_actions: int, n_agents: int,
                 lr: float = 1e-3, device: str = 'cuda'):
        self.obs_size  = obs_size
        self.obs_dim   = obs_size * obs_size
        self.n_actions = n_actions
        self.n_agents  = n_agents
        self.device    = device

        self.forward_model  = ForwardModel(self.obs_dim, n_actions).to(device)
        self.inverse_model  = InverseModel(self.obs_dim, n_actions).to(device)

        self.fm_optimizer = optim.Adam(self.forward_model.parameters(), lr=lr)
        self.im_optimizer = optim.Adam(self.inverse_model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Grounded-action inference  (used during fine-tuning)
    # ------------------------------------------------------------------

    def ground_action(self, obs: np.ndarray, intended_action: int) -> int:
        """
        Given the agent's intended action (from DQN policy), return the
        grounded action that accounts for the sim/real dynamics gap.

        Steps:
          1. ForwardModel predicts where the agent would end up in the *real* env
             (forward model was trained on real trajectories).
          2. InverseModel maps (obs, real_predicted_next_obs) → grounded action.
             Since the inverse model was trained on sim transitions, it will output
             the sim action that best reproduces the real-dynamics next state.
        """
        self.forward_model.eval()
        self.inverse_model.eval()
        with torch.no_grad():
            obs_t    = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(self.device)
            a_onehot = F.one_hot(
                torch.tensor([intended_action]), num_classes=self.n_actions
            ).float().to(self.device)

            # Forward model predicts where real dynamics would take us
            pred_real_next = self.forward_model(obs_t, a_onehot)      # (1, obs_dim)

            # Inverse model asks: which sim action leads to this next state?
            logits   = self.inverse_model(obs_t, pred_real_next)       # (1, n_actions)
            grounded = logits.argmax(dim=-1).item()
        return grounded

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def train_forward_step(self, batch):
        """
        One gradient step on the forward model using a batch of *real* transitions.
        Teaches the forward model P_real(s' | s, a).

        batch: (obs, actions, next_obs)  – numpy arrays, collected from real env
        """
        self.forward_model.train()
        obs, actions, next_obs = batch
        obs_t      = torch.FloatTensor(obs).to(self.device)
        a_onehot   = F.one_hot(
            torch.LongTensor(actions).to(self.device), num_classes=self.n_actions
        ).float()
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)

        pred_next = self.forward_model(obs_t, a_onehot)
        loss = F.mse_loss(pred_next, next_obs_t)

        self.fm_optimizer.zero_grad()
        loss.backward()
        self.fm_optimizer.step()
        return loss.item()

    def train_inverse_step(self, batch):
        """
        One gradient step on the inverse model using a batch of *sim* transitions.
        Teaches the inverse model π_inv(a | s, s') using only sim data.

        The inverse model learns: given (obs_t, next_obs_t) from sim, predict the
        action that caused that transition. At inference time, next_obs_t is replaced
        by the forward model's real-dynamics prediction, so the inverse model
        effectively outputs the sim action that best replicates real-world outcomes.

        batch: (obs, actions, next_obs)  - numpy arrays, collected from sim env
        """
        self.inverse_model.train()
        obs, actions, next_obs = batch
        obs_t      = torch.FloatTensor(obs).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)

        # Directly supervised: given the actual sim transition (obs → next_obs),
        # predict the action that caused it.
        logits = self.inverse_model(obs_t, next_obs_t)
        loss   = F.cross_entropy(logits, torch.LongTensor(actions).to(self.device))

        self.im_optimizer.zero_grad()
        loss.backward()
        self.im_optimizer.step()
        return loss.item()

    def save(self, path: str):
        torch.save({
            'forward_model': self.forward_model.state_dict(),
            'inverse_model': self.inverse_model.state_dict(),
            'fm_optimizer':  self.fm_optimizer.state_dict(),
            'im_optimizer':  self.im_optimizer.state_dict(),
        }, path)
        print(f"[SharedGAT] Saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.forward_model.load_state_dict(ckpt['forward_model'])
        self.inverse_model.load_state_dict(ckpt['inverse_model'])
        self.fm_optimizer.load_state_dict(ckpt['fm_optimizer'])
        self.im_optimizer.load_state_dict(ckpt['im_optimizer'])
        print(f"[SharedGAT] Loaded from {path}")


class PerAgentGAT:
    """
    Each agent gets its own forward model and inverse model (no parameter sharing).
    """

    def __init__(self, obs_size: int, n_actions: int, n_agents: int,
                 lr: float = 1e-3, device: str = 'cuda'):
        self.obs_size  = obs_size
        self.obs_dim   = obs_size * obs_size
        self.n_actions = n_actions
        self.n_agents  = n_agents
        self.device    = device

        self.forward_models = nn.ModuleList([
            ForwardModel(self.obs_dim, n_actions).to(device)
            for _ in range(n_agents)
        ])
        self.inverse_models = nn.ModuleList([
            InverseModel(self.obs_dim, n_actions).to(device)
            for _ in range(n_agents)
        ])

        self.fm_optimizers = [
            optim.Adam(fm.parameters(), lr=lr)
            for fm in self.forward_models
        ]
        self.im_optimizers = [
            optim.Adam(im.parameters(), lr=lr)
            for im in self.inverse_models
        ]

    def ground_action(self, agent_id: int, obs: np.ndarray, intended_action: int) -> int:
        fm = self.forward_models[agent_id]
        im = self.inverse_models[agent_id]
        fm.eval(); im.eval()
        with torch.no_grad():
            obs_t    = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(self.device)
            a_onehot = F.one_hot(
                torch.tensor([intended_action]), num_classes=self.n_actions
            ).float().to(self.device)

            # Forward model predicts where real dynamics would take us
            pred_real_next = fm(obs_t, a_onehot)

            # Inverse model asks: which sim action leads to this next state?
            logits   = im(obs_t, pred_real_next)
            grounded = logits.argmax(dim=-1).item()
        return grounded

    def train_forward_step(self, agent_id: int, batch):
        obs, actions, next_obs = batch
        fm  = self.forward_models[agent_id]
        opt = self.fm_optimizers[agent_id]
        fm.train()

        obs_t      = torch.FloatTensor(obs).to(self.device)
        a_onehot   = F.one_hot(
            torch.LongTensor(actions).to(self.device), num_classes=self.n_actions
        ).float()
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)

        pred_next = fm(obs_t, a_onehot)
        loss = F.mse_loss(pred_next, next_obs_t)

        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    def train_inverse_step(self, agent_id: int, batch):
        """
        Train purely on sim (obs, action, next_obs) triples.
        No forward model involvement — that only happens at inference.
        """
        obs, actions, next_obs = batch
        im  = self.inverse_models[agent_id]
        opt = self.im_optimizers[agent_id]
        im.train()

        obs_t      = torch.FloatTensor(obs).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)

        # Supervised on actual sim transitions: (obs, next_obs) → action
        logits = im(obs_t, next_obs_t)
        loss   = F.cross_entropy(logits, torch.LongTensor(actions).to(self.device))

        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    def save(self, path: str):
        state = {
            f'forward_model_{i}': self.forward_models[i].state_dict()
            for i in range(self.n_agents)
        }
        state.update({
            f'inverse_model_{i}': self.inverse_models[i].state_dict()
            for i in range(self.n_agents)
        })
        state.update({
            f'fm_optimizer_{i}': self.fm_optimizers[i].state_dict()
            for i in range(self.n_agents)
        })
        state.update({
            f'im_optimizer_{i}': self.im_optimizers[i].state_dict()
            for i in range(self.n_agents)
        })
        torch.save(state, path)
        print(f"[PerAgentGAT] Saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        for i in range(self.n_agents):
            self.forward_models[i].load_state_dict(ckpt[f'forward_model_{i}'])
            self.inverse_models[i].load_state_dict(ckpt[f'inverse_model_{i}'])
            self.fm_optimizers[i].load_state_dict(ckpt[f'fm_optimizer_{i}'])
            self.im_optimizers[i].load_state_dict(ckpt[f'im_optimizer_{i}'])
        print(f"[PerAgentGAT] Loaded from {path}")


# ---------------------------------------------------------------------------
# Trajectory Collection
# ---------------------------------------------------------------------------

class TransitionBuffer:
    """Lightweight buffer storing (obs, action, next_obs) triples for GAT training."""

    def __init__(self):
        self._data = []   # list of (obs_flat, action, next_obs_flat)

    def push(self, obs_flat, action, next_obs_flat):
        self._data.append((obs_flat, action, next_obs_flat))

    def sample(self, batch_size: int):
        batch = random.sample(self._data, min(batch_size, len(self._data)))
        obs, actions, next_obs = zip(*batch)
        return np.array(obs), np.array(actions), np.array(next_obs)

    def __len__(self):
        return len(self._data)


def collect_trajectories(
    env_config: dict,
    n_episodes: int,
    obs_size: int,
    policy: IndependentDQN | None = None,
    epsilon: float = 1.0,
    seed: int = 22,
    label: str = "env",
) -> list[TransitionBuffer]:
    """
    Collect transition data from an environment.

    Returns a list of n_agents TransitionBuffers, one per agent.
    """
    env      = ExplorerMALocalObs(conf=env_config)
    n_agents = env.n_agents
    buffers  = [TransitionBuffer() for _ in range(n_agents)]

    print(f"\n[Collect] {label}: {n_episodes} episodes | "
          f"mode={env_config.get('env_mode')} | slip={env_config.get('slip_prob', 0.0)}")

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed)
        done   = False
        steps  = 0

        while not done:
            if policy is not None and random.random() > epsilon:
                actions = [policy.select_action(obs[i], eval_mode=True) for i in range(n_agents)]
            else:
                actions = [random.randrange(4) for _ in range(n_agents)]

            next_obs, _, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated

            for i in range(n_agents):
                buffers[i].push(
                    obs[i].flatten().astype(np.float32),
                    actions[i],
                    next_obs[i].flatten().astype(np.float32),
                )

            obs   = next_obs
            steps += 1

        if (ep + 1) % max(1, n_episodes // 5) == 0:
            total = sum(len(b) for b in buffers)
            print(f"  Episode {ep+1}/{n_episodes} | total transitions={total}")

    env.close()
    return buffers


# ---------------------------------------------------------------------------
# GAT Training Loop
# ---------------------------------------------------------------------------

def train_gat_models(
    gat,
    real_buffers:  list[TransitionBuffer],
    sim_buffers:   list[TransitionBuffer],
    n_epochs:      int   = 50,
    batch_size:    int   = 256,
    shared:        bool  = True,
) -> dict:
    """
    Train forward models on real data and inverse models on sim data.

    Forward model: real (obs, action) → next_obs   [MSE loss]
    Inverse model: sim  (obs, next_obs) → action    [cross-entropy loss]
    """
    n_agents = len(real_buffers)
    history  = {'fm_losses': [], 'im_losses': []}

    print(f"\n{'='*60}")
    print(f"Training GAT models  ({'shared' if shared else 'per-agent'})")
    print(f"  Forward model  → {sum(len(b) for b in real_buffers)} real transitions")
    print(f"  Inverse model  → {sum(len(b) for b in sim_buffers)} sim transitions")
    print(f"  Epochs: {n_epochs} | Batch: {batch_size}")
    print(f"{'='*60}")

    for epoch in range(n_epochs):
        epoch_fm = []
        epoch_im = []

        if shared:
            all_real_obs, all_real_act, all_real_next = [], [], []
            all_sim_obs,  all_sim_act,  all_sim_next  = [], [], []

            for i in range(n_agents):
                ro, ra, rn = real_buffers[i].sample(batch_size // n_agents)
                so, sa, sn = sim_buffers[i].sample(batch_size // n_agents)
                all_real_obs.append(ro); all_real_act.append(ra); all_real_next.append(rn)
                all_sim_obs.append(so);  all_sim_act.append(sa);  all_sim_next.append(sn)

            real_batch = (
                np.concatenate(all_real_obs),
                np.concatenate(all_real_act),
                np.concatenate(all_real_next),
            )
            sim_batch  = (
                np.concatenate(all_sim_obs),
                np.concatenate(all_sim_act),
                np.concatenate(all_sim_next),
            )

            fm_loss = gat.train_forward_step(real_batch)
            im_loss = gat.train_inverse_step(sim_batch)
            epoch_fm.append(fm_loss)
            epoch_im.append(im_loss)

        else:
            for i in range(n_agents):
                real_batch = real_buffers[i].sample(batch_size)
                sim_batch  = sim_buffers[i].sample(batch_size)
                fm_loss    = gat.train_forward_step(i, real_batch)
                im_loss    = gat.train_inverse_step(i, sim_batch)
                epoch_fm.append(fm_loss)
                epoch_im.append(im_loss)

        history['fm_losses'].append(float(np.mean(epoch_fm)))
        history['im_losses'].append(float(np.mean(epoch_im)))

        if (epoch + 1) % max(1, n_epochs // 10) == 0:
            print(f"  Epoch {epoch+1:>4}/{n_epochs} | "
                  f"FM loss: {history['fm_losses'][-1]:.5f} | "
                  f"IM loss: {history['im_losses'][-1]:.5f}")

    print("GAT training complete.\n")
    return history


# ---------------------------------------------------------------------------
# Policy Fine-Tuning with GAT
# ---------------------------------------------------------------------------

def finetune_with_gat(
    base_agent_path: str,
    gat,
    env_config:      dict,
    env_config_real: dict,
    obs_size:        int   = 7,
    n_episodes:      int   = 500,
    shared:          bool  = True,
    save_freq:       int   = 100,
    log_freq:        int   = 10,
    eval_freq:       int   = 10,
    n_eval_episodes: int   = 10,
    checkpoint_dir:  str   = 'checkpoints',
    device:          str   = 'cuda',
    seed:            int   = 22,
) -> IndependentDQN:
    """
    Fine-tune a pre-trained DQN policy in simulation using GAT-grounded actions.

    Rollouts are collected in sim using grounded actions. Evaluation is performed
    in the real environment (with slip) averaged over n_eval_episodes so the best
    checkpoint reflects genuine sim-to-real transfer. Fine-tuning is fully greedy
    (epsilon=0).
    """
    variant = 'shared' if shared else 'per_agent'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(checkpoint_dir, f'gat_{variant}_finetune_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Fine-tuning DQN with GAT  (variant={variant})")
    print(f"  Base policy    : {base_agent_path}")
    print(f"  Episodes       : {n_episodes}")
    print(f"  Exploration    : greedy (epsilon=0)")
    print(f"  Eval env       : real (slip_prob={env_config_real.get('slip_prob', '?')})")
    print(f"  Eval episodes  : {n_eval_episodes} (averaged)")
    print(f"  Run dir        : {run_dir}")
    print(f"{'='*60}\n")

    # Load base policy — fully greedy, no epsilon exploration
    agent = IndependentDQN(obs_size=obs_size, n_actions=4, device=device)
    agent.load(base_agent_path)
    agent.epsilon       = 0.0
    agent.epsilon_end   = 0.0
    agent.epsilon_decay = 1.0

    env      = ExplorerMALocalObs(conf=env_config)
    n_agents = env.n_agents

    ep_rewards, ep_coverages = [], []
    losses = []
    eval_rewards_hist, eval_coverages_hist, eval_ep_nums = [], [], []
    best_eval_reward = -float('inf')

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed)
        ep_reward = [0.0] * n_agents
        done      = False

        while not done:
            # 1. DQN selects intended actions (greedy)
            intended = [agent.select_action(obs[i]) for i in range(n_agents)]

            # 2. GAT grounds each intended action via forward model → inverse model
            if shared:
                grounded = [gat.ground_action(obs[i], intended[i]) for i in range(n_agents)]
            else:
                grounded = [gat.ground_action(i, obs[i], intended[i]) for i in range(n_agents)]

            # 3. Step sim with grounded actions
            next_obs, rewards, terminated, truncated, _ = env.step(grounded)
            done = terminated or truncated

            # 4. Store grounded action — the policy must learn the value of the
            #    action that was actually executed in the environment.
            for i in range(n_agents):
                #agent.store_transition(obs[i], grounded[i], rewards[i], next_obs[i], done)
                agent.store_transition(obs[i], intended[i], rewards[i], next_obs[i], done)
                ep_reward[i] += rewards[i]

            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            obs = next_obs

        total_cells    = env.SIZE[0] * env.SIZE[1]
        explored_cells = np.count_nonzero(env.exploredMap)
        coverage       = explored_cells / total_cells

        ep_rewards.append(float(np.mean(ep_reward)))
        ep_coverages.append(coverage)

        # Deterministic evaluation averaged over n_eval_episodes in the REAL environment
        if (episode + 1) % eval_freq == 0:
            eval_r, eval_c = _eval_episodes(
                env_config_real, agent, obs_size, seed, device, n_eval_episodes
            )
            eval_rewards_hist.append(eval_r)
            eval_coverages_hist.append(eval_c)
            eval_ep_nums.append(episode + 1)

            print(f"\n[EVAL/REAL] Episode {episode+1} | "
                  f"Reward={eval_r:.2f} | Coverage={eval_c:.2%} "
                  f"(avg over {n_eval_episodes} episodes)")

            if eval_r > best_eval_reward:
                best_eval_reward = eval_r
                best_path = os.path.join(run_dir, 'best_model.pt')
                agent.save(best_path)
                print(f"  *** New best real-env model saved: {eval_r:.2f} ***")

        if (episode + 1) % log_freq == 0:
            recent_r = np.mean(ep_rewards[-log_freq:])
            recent_c = np.mean(ep_coverages[-log_freq:])
            avg_loss = np.mean(losses[-200:]) if losses else 0.0
            print(f"EP {episode+1:>5}/{n_episodes} | "
                  f"Reward={recent_r:.2f} | Coverage={recent_c:.2%} | "
                  f"Loss={avg_loss:.4f}")

        if (episode + 1) % save_freq == 0:
            ckpt_path = os.path.join(run_dir, f'checkpoint_ep{episode+1}.pt')
            agent.save(ckpt_path)

    agent.save(os.path.join(run_dir, 'final_model.pt'))

    metrics = {
        'ep_rewards':       ep_rewards,
        'ep_coverages':     ep_coverages,
        'losses':           losses,
        'eval_rewards':     eval_rewards_hist,
        'eval_coverages':   eval_coverages_hist,
        'eval_episodes':    eval_ep_nums,
        'best_eval_reward': best_eval_reward,
        'variant':          variant,
        'eval_env':         'real',
        'slip_prob':        env_config_real.get('slip_prob'),
        'n_eval_episodes':  n_eval_episodes,
    }
    np.save(os.path.join(run_dir, 'metrics.npy'), metrics)

    print(f"\nFine-tuning done. Best real-env reward: {best_eval_reward:.2f}")
    print(f"Run directory: {run_dir}")
    return agent


def _eval_episodes(env_config, agent, obs_size, seed, device, n_episodes: int = 10):
    """
    Run n_episodes deterministic evaluation episodes and return averaged
    (reward, coverage) over all episodes.

    Each episode uses the same seed.
    """
    all_rewards   = []
    all_coverages = []

    for i in range(n_episodes):
        env      = ExplorerMALocalObs(conf=env_config)
        n_agents = env.n_agents
        obs, _   = env.reset(seed=seed)
        ep_reward = [0.0] * n_agents
        done      = False

        while not done:
            actions  = [agent.select_action(obs[j], eval_mode=True) for j in range(n_agents)]
            next_obs, rewards, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            for j in range(n_agents):
                ep_reward[j] += rewards[j]
            obs = next_obs

        total_cells    = env.SIZE[0] * env.SIZE[1]
        explored_cells = np.count_nonzero(env.exploredMap)
        env.close()

        all_rewards.append(float(np.mean(ep_reward)))
        all_coverages.append(explored_cells / total_cells)

    return float(np.mean(all_rewards)), float(np.mean(all_coverages))


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def run_gat_pipeline(
    base_agent_path:      str,
    env_config_sim:       dict,
    env_config_real:      dict,
    variant:              str  = 'shared',
    obs_size:             int  = 7,
    n_agents:             int  = 2,
    n_actions:            int  = 4,
    collect_episodes_real: int = 200,
    collect_episodes_sim:  int = 200,
    gat_epochs:           int  = 100,
    gat_batch_size:       int  = 256,
    gat_lr:               float = 1e-3,
    finetune_episodes:    int  = 500,
    device:               str  = 'cuda',
    checkpoint_dir:       str  = 'checkpoints',
    seed:                 int  = 22,
    save_freq:            int  = 100,
    log_freq:             int  = 10,
    eval_freq:            int  = 10,
    n_eval_episodes:      int  = 10,
):
    """
    End-to-end GAT pipeline:
      1. Load pre-trained base policy
      2. Collect real-world trajectories  (used to train forward model)
      3. Collect simulation trajectories  (used to train inverse model)
      4. Train GAT (forward + inverse) models
      5. Fine-tune DQN policy in sim using GAT-grounded actions,
         evaluated against the real environment (averaged over n_eval_episodes)
    """
    shared = (variant == 'shared')
    assert variant in ('shared', 'per_agent'), "variant must be 'shared' or 'per_agent'"

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_root  = os.path.join(checkpoint_dir, f'gat_{variant}_{timestamp}')
    os.makedirs(run_root, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  MA-GAT PIPELINE  |  variant={variant}")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Step 1 – Load base policy for guided collection                      #
    # ------------------------------------------------------------------ #
    print("\n[Step 1] Loading base policy for trajectory collection …")
    base_agent = IndependentDQN(obs_size=obs_size, n_actions=n_actions, device=device)
    base_agent.load(base_agent_path)

    # ------------------------------------------------------------------ #
    # Step 2 – Collect real-world trajectories                             #
    # ------------------------------------------------------------------ #
    print("\n[Step 2] Collecting real-world trajectories …")
    real_buffers = collect_trajectories(
        env_config=env_config_real,
        n_episodes=collect_episodes_real,
        obs_size=obs_size,
        policy=base_agent,
        epsilon=0.3,
        seed=seed,
        label="REAL",
    )
    print(f"  Real buffer sizes: {[len(b) for b in real_buffers]}")

    # ------------------------------------------------------------------ #
    # Step 3 – Collect simulation trajectories                             #
    # ------------------------------------------------------------------ #
    print("\n[Step 3] Collecting simulation trajectories …")
    sim_buffers = collect_trajectories(
        env_config=env_config_sim,
        n_episodes=collect_episodes_sim,
        obs_size=obs_size,
        policy=base_agent,
        epsilon=0.3,
        seed=seed,
        label="SIM",
    )
    print(f"  Sim buffer sizes: {[len(b) for b in sim_buffers]}")

    # ------------------------------------------------------------------ #
    # Step 4 – Build and train GAT models                                  #
    # ------------------------------------------------------------------ #
    print("\n[Step 4] Building and training GAT models …")
    if shared:
        gat = SharedGAT(obs_size, n_actions, n_agents, lr=gat_lr, device=device)
    else:
        gat = PerAgentGAT(obs_size, n_actions, n_agents, lr=gat_lr, device=device)

    gat_history = train_gat_models(
        gat=gat,
        real_buffers=real_buffers,
        sim_buffers=sim_buffers,
        n_epochs=gat_epochs,
        batch_size=gat_batch_size,
        shared=shared,
    )

    gat_save_path = os.path.join(run_root, 'gat_models.pt')
    gat.save(gat_save_path)

    with open(os.path.join(run_root, 'gat_training_history.json'), 'w') as f:
        json.dump(gat_history, f, indent=2)

    # ------------------------------------------------------------------ #
    # Step 5 – Fine-tune DQN policy in sim with GAT                        #
    # ------------------------------------------------------------------ #
    print("\n[Step 5] Fine-tuning DQN policy with GAT …")
    finetuned_agent = finetune_with_gat(
        base_agent_path=base_agent_path,
        gat=gat,
        env_config=env_config_sim,
        env_config_real=env_config_real,
        obs_size=obs_size,
        n_episodes=finetune_episodes,
        shared=shared,
        save_freq=save_freq,
        log_freq=log_freq,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        checkpoint_dir=run_root,
        device=device,
        seed=seed,
    )

    print("\n" + "=" * 70)
    print(f"  MA-GAT PIPELINE COMPLETE  |  variant={variant}")
    print(f"  All outputs saved to: {run_root}")
    print("=" * 70)

    return finetuned_agent, gat


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-Agent Grounded Action Transformation (MA-GAT)'
    )
    parser.add_argument('--base_model', type=str, required=True,
                        help='Path to pre-trained IndependentDQN checkpoint')
    parser.add_argument('--variant', type=str, default='shared',
                        choices=['shared', 'per_agent'],
                        help='GAT variant: shared (parameter sharing) or per_agent')
    parser.add_argument('--slip_prob', type=float, default=0.3,
                        help='Slip probability for real environment (default: 0.3)')
    parser.add_argument('--collect_real', type=int, default=200,
                        help='Real-world collection episodes (default: 200)')
    parser.add_argument('--collect_sim', type=int, default=200,
                        help='Simulation collection episodes (default: 200)')
    parser.add_argument('--gat_epochs', type=int, default=100,
                        help='GAT training epochs (default: 100)')
    parser.add_argument('--gat_lr', type=float, default=1e-3,
                        help='GAT learning rate (default: 1e-3)')
    parser.add_argument('--finetune_episodes', type=int, default=500,
                        help='DQN fine-tuning episodes (default: 500)')
    parser.add_argument('--n_eval_episodes', type=int, default=10,
                        help='Real-env eval episodes to average over (default: 10)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=22,
                        help='Environment / map seed (default: 22)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Root checkpoint directory (default: checkpoints)')
    args = parser.parse_args()

    # ── Environment configuration ─────────────────────────────────────────
    conf["n_agents"]             = 2
    conf["shared_map"]           = True
    conf["size"]                 = [30, 30]
    conf["obstacles"]            = 10
    conf["lidar_range"]          = 2
    conf["obstacle_size"]        = [1, 3]
    conf["initial"]              = [1, 1]
    conf["collision_reward"]     = -50
    conf["out_of_bounds_reward"] = -50
    conf["movementCost"]         = 0.1
    conf["max_steps"]            = 200
    conf["verbose_slip"]         = False

    env_config_sim = conf.copy()
    env_config_sim['env_mode']  = 'sim'
    env_config_sim['slip_prob'] = 0.0

    env_config_real = conf.copy()
    env_config_real['env_mode']  = 'real'
    env_config_real['slip_prob'] = args.slip_prob

    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'

    run_gat_pipeline(
        base_agent_path=args.base_model,
        env_config_sim=env_config_sim,
        env_config_real=env_config_real,
        variant=args.variant,
        obs_size=7,
        n_agents=conf["n_agents"],
        n_actions=4,
        collect_episodes_real=args.collect_real,
        collect_episodes_sim=args.collect_sim,
        gat_epochs=args.gat_epochs,
        gat_lr=args.gat_lr,
        finetune_episodes=args.finetune_episodes,
        n_eval_episodes=args.n_eval_episodes,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )