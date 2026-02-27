"""
Multi-Agent Grounded Action Transformation (MA-GAT) for Sim-to-Real Transfer
with optional Stochastic GAT (SGAT) support.

Two training modes:
  --mode finetune   : Load a pre-trained IDQN checkpoint, finetune with GAT/SGAT
  --mode scratch    : Train IDQN from scratch inside the GAT/SGAT loop

Two GAT variants:
  --variant shared     : All agents share one forward model + one inverse model
  --variant per_agent  : Each agent has its own forward + inverse model

Stochasticity flag:
  --sgat            : Use Stochastic GAT (SGAT). The forward model outputs a
                      Gaussian distribution (mean + log_std) and samples from it
                      during grounding, rather than predicting a single next state.
                      This captures real-world stochasticity that deterministic GAT
                      misses.

Key fixes vs previous version:
  1. From-scratch mode trains IDQN with epsilon exploration inside the GAT loop
  2. Replay buffer stores GROUNDED actions (what was actually executed), not intended
  3. Best model selected by real-environment validation coverage (not reward)
  4. Finetuning mode retains small epsilon for stability
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

from mars_explorer.envs.explorer import ExplorerMALocalObs
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf
from train_idqn import DQN_CNN, IndependentDQN, ReplayBuffer


# ---------------------------------------------------------------------------
# GAT Neural-Network Components
# ---------------------------------------------------------------------------

class ForwardModel(nn.Module):
    """
    Deterministic forward model. Predicts next observation under "real" environment dynamics.

    Input:  flat(obs_t) || one_hot(action_t)    dim = obs_dim + n_actions
    Output: flat(obs_{t+1})                      dim = obs_dim

    Trained with MSE loss on real-world trajectories.
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
        return self.net(torch.cat([obs, action_onehot], dim=-1))


class StochasticForwardModel(nn.Module):
    """
    Stochastic forward model for SGAT. Predicts a *distribution* over next
    observations under "real" environment dynamics instead of a single point estimate.

    Architecture:
      - A shared trunk encodes (obs_t, action_t)
      - Two output heads produce mean and log_std for each obs dimension
      - At inference, a next state is sampled: next_obs ~ N(mean, std^2)

    Input:  flat(obs_t) || one_hot(action_t)              dim = obs_dim + n_actions
    Output: (mean, log_std) each of shape (batch, obs_dim)

    Trained with Gaussian NLL loss on real-world trajectories:
        L = 0.5 * sum[ log(std^2) + (next_obs - mean)^2 / std^2 ]
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256,
                 log_std_min: float = -5.0, log_std_max: float = 2.0):
        super().__init__()
        self.obs_dim     = obs_dim
        self.n_actions   = n_actions
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, obs_dim)
        self.log_std_head = nn.Linear(hidden_dim, obs_dim)

    def forward(self, obs: torch.Tensor,
                action_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h       = self.trunk(torch.cat([obs, action_onehot], dim=-1))
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: torch.Tensor,
               action_onehot: torch.Tensor) -> torch.Tensor:
        """Sample a next observation from the predicted Gaussian distribution."""
        mean, log_std = self(obs, action_onehot)
        std           = log_std.exp()
        eps           = torch.randn_like(mean)
        return mean + std * eps     # reparameterised sample


class InverseModel(nn.Module):
    """
    Maps (obs_t, predicted_real_next_obs) → grounded sim action.

    Input:  flat(obs_t) || flat(obs_{t+1})    dim = 2 * obs_dim
    Output: logits over actions                dim = n_actions

    Trained on SIM (obs, action, next_obs) triples using cross-entropy.
    At inference, next_obs is replaced by the forward model's real-dynamics
    prediction (deterministic GAT) or sample (SGAT), so the inverse model
    outputs the sim action that best replicates that real-dynamics outcome.
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
        return self.net(torch.cat([obs, next_obs], dim=-1))


# ---------------------------------------------------------------------------
# Gaussian NLL Loss Helper
# ---------------------------------------------------------------------------

def gaussian_nll_loss(mean: torch.Tensor, log_std: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
    """
    Element-wise Gaussian NLL, averaged over batch and state dimensions.

    L = 0.5 * mean_over_dims[ 2*log_std + (target - mean)^2 / exp(2*log_std) ]

    This is equivalent to -log p(target | mean, std) under N(mean, std^2),
    up to a constant. Minimizing this teaches the model the correct spread as
    well as the correct centre.
    """
    var  = (2 * log_std).exp()                          # std^2
    loss = 0.5 * (2 * log_std + (target - mean).pow(2) / var)
    return loss.mean()


# ---------------------------------------------------------------------------
# GAT Agent Wrappers  (Shared)
# ---------------------------------------------------------------------------

class SharedGAT:
    """
    Deterministic GAT — single forward + inverse model shared across all agents.

    ground_action: obs + intended_action
        forward model predicts single next state
        inverse model selects grounded action
    """

    def __init__(self, obs_size: int, n_actions: int, n_agents: int,
                 lr: float = 1e-3, device: str = 'cuda'):
        self.obs_size  = obs_size
        self.obs_dim   = obs_size * obs_size
        self.n_actions = n_actions
        self.n_agents  = n_agents
        self.device    = device

        self.forward_model = ForwardModel(self.obs_dim, n_actions).to(device)
        self.inverse_model = InverseModel(self.obs_dim, n_actions).to(device)
        self.fm_optimizer  = optim.Adam(self.forward_model.parameters(), lr=lr)
        self.im_optimizer  = optim.Adam(self.inverse_model.parameters(), lr=lr)

    def ground_action(self, obs: np.ndarray, intended_action: int) -> int:
        """
        Transform intended action → grounded action via forward + inverse models.

        1. Forward model predicts where "real" environment dynamics would take the agent.
        2. Inverse model asks: which SIM action produces that same next state?
        That sim action is the grounded action — it compensates for the gap.
        """
        self.forward_model.eval()
        self.inverse_model.eval()
        with torch.no_grad():
            obs_t    = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(self.device)
            a_onehot = F.one_hot(
                torch.tensor([intended_action]), num_classes=self.n_actions
            ).float().to(self.device)
            pred_real_next = self.forward_model(obs_t, a_onehot)
            logits         = self.inverse_model(obs_t, pred_real_next)
            return logits.argmax(dim=-1).item()

    def train_forward_step(self, batch):
        """MSE on real transitions: (obs, action) → next_obs."""
        self.forward_model.train()
        obs, actions, next_obs = batch
        obs_t      = torch.FloatTensor(obs).to(self.device)
        a_onehot   = F.one_hot(
            torch.LongTensor(actions).to(self.device), num_classes=self.n_actions
        ).float()
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        loss = F.mse_loss(self.forward_model(obs_t, a_onehot), next_obs_t)
        self.fm_optimizer.zero_grad(); loss.backward(); self.fm_optimizer.step()
        return loss.item()

    def train_inverse_step(self, batch):
        """Cross-entropy on sim transitions: (obs, next_obs) → action."""
        self.inverse_model.train()
        obs, actions, next_obs = batch
        obs_t      = torch.FloatTensor(obs).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        logits = self.inverse_model(obs_t, next_obs_t)
        loss   = F.cross_entropy(logits, torch.LongTensor(actions).to(self.device))
        self.im_optimizer.zero_grad(); loss.backward(); self.im_optimizer.step()
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


class SharedSGAT(SharedGAT):
    """
    Stochastic GAT — single stochastic forward model + inverse model shared
    across all agents.

    The only difference from SharedGAT:
      1. forward_model is a StochasticForwardModel (outputs mean + log_std)
      2. train_forward_step uses Gaussian NLL instead of MSE
      3. ground_action SAMPLES from the predicted distribution instead of
         taking the argmax / point prediction

    By sampling, the grounded simulator inherits the real world's stochastic
    transition structure, allowing the policy to learn risk-averse behaviour
    that pure deterministic GAT cannot capture.
    """

    def __init__(self, obs_size: int, n_actions: int, n_agents: int,
                 lr: float = 1e-3, device: str = 'cuda'):
        # Initialise base class to get inverse model + optimiser boilerplate
        super().__init__(obs_size, n_actions, n_agents, lr, device)

        # Replace deterministic forward model with stochastic version
        self.forward_model = StochasticForwardModel(self.obs_dim, n_actions).to(device)
        self.fm_optimizer  = optim.Adam(self.forward_model.parameters(), lr=lr)

    # --- Override: sample instead of point-predict ---------------------------
    def ground_action(self, obs: np.ndarray, intended_action: int) -> int:
        """
        Grounding with stochastic forward model:
          1. Sample next_obs ~ N(mean, std^2) predicted by stochastic FM
          2. Inverse model selects grounded action for that sampled next_obs

        Because the sample varies each call, the grounded simulator is itself
        stochastic — matching the stochasticity of the real environment and
        enabling the downstream policy to learn appropriate risk-sensitive behaviour.
        """
        self.forward_model.eval()
        self.inverse_model.eval()
        with torch.no_grad():
            obs_t    = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(self.device)
            a_onehot = F.one_hot(
                torch.tensor([intended_action]), num_classes=self.n_actions
            ).float().to(self.device)
            # KEY DIFFERENCE vs GAT: sample from predicted distribution
            sampled_next = self.forward_model.sample(obs_t, a_onehot)
            logits       = self.inverse_model(obs_t, sampled_next)
            return logits.argmax(dim=-1).item()

    # --- Override: NLL loss instead of MSE -----------------------------------
    def train_forward_step(self, batch):
        """
        Gaussian NLL on real transitions: (obs, action) → distribution over next_obs.

        Unlike MSE, NLL penalises both incorrect means and incorrect variances,
        so the model learns a calibrated uncertainty over next states.
        """
        self.forward_model.train()
        obs, actions, next_obs = batch
        obs_t      = torch.FloatTensor(obs).to(self.device)
        a_onehot   = F.one_hot(
            torch.LongTensor(actions).to(self.device), num_classes=self.n_actions
        ).float()
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)

        mean, log_std = self.forward_model(obs_t, a_onehot)
        loss = gaussian_nll_loss(mean, log_std, next_obs_t)
        self.fm_optimizer.zero_grad(); loss.backward(); self.fm_optimizer.step()
        return loss.item()

    def save(self, path: str):
        torch.save({
            'forward_model': self.forward_model.state_dict(),
            'inverse_model': self.inverse_model.state_dict(),
            'fm_optimizer':  self.fm_optimizer.state_dict(),
            'im_optimizer':  self.im_optimizer.state_dict(),
        }, path)
        print(f"[SharedSGAT] Saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.forward_model.load_state_dict(ckpt['forward_model'])
        self.inverse_model.load_state_dict(ckpt['inverse_model'])
        self.fm_optimizer.load_state_dict(ckpt['fm_optimizer'])
        self.im_optimizer.load_state_dict(ckpt['im_optimizer'])
        print(f"[SharedSGAT] Loaded from {path}")


# ---------------------------------------------------------------------------
# GAT Agent Wrappers  (Per-Agent)
# ---------------------------------------------------------------------------

class PerAgentGAT:
    """Each agent has its own deterministic forward + inverse model."""

    def __init__(self, obs_size: int, n_actions: int, n_agents: int,
                 lr: float = 1e-3, device: str = 'cuda'):
        self.obs_size  = obs_size
        self.obs_dim   = obs_size * obs_size
        self.n_actions = n_actions
        self.n_agents  = n_agents
        self.device    = device

        self.forward_models = nn.ModuleList([
            ForwardModel(self.obs_dim, n_actions).to(device) for _ in range(n_agents)
        ])
        self.inverse_models = nn.ModuleList([
            InverseModel(self.obs_dim, n_actions).to(device) for _ in range(n_agents)
        ])
        self.fm_optimizers = [optim.Adam(fm.parameters(), lr=lr) for fm in self.forward_models]
        self.im_optimizers = [optim.Adam(im.parameters(), lr=lr) for im in self.inverse_models]

    def ground_action(self, agent_id: int, obs: np.ndarray, intended_action: int) -> int:
        fm = self.forward_models[agent_id]
        im = self.inverse_models[agent_id]
        fm.eval(); im.eval()
        with torch.no_grad():
            obs_t    = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(self.device)
            a_onehot = F.one_hot(
                torch.tensor([intended_action]), num_classes=self.n_actions
            ).float().to(self.device)
            pred_real_next = fm(obs_t, a_onehot)
            logits         = im(obs_t, pred_real_next)
            return logits.argmax(dim=-1).item()

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
        loss = F.mse_loss(fm(obs_t, a_onehot), next_obs_t)
        opt.zero_grad(); loss.backward(); opt.step()
        return loss.item()

    def train_inverse_step(self, agent_id: int, batch):
        obs, actions, next_obs = batch
        im  = self.inverse_models[agent_id]
        opt = self.im_optimizers[agent_id]
        im.train()
        obs_t      = torch.FloatTensor(obs).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        logits = im(obs_t, next_obs_t)
        loss   = F.cross_entropy(logits, torch.LongTensor(actions).to(self.device))
        opt.zero_grad(); loss.backward(); opt.step()
        return loss.item()

    def save(self, path: str):
        state = {}
        for i in range(self.n_agents):
            state[f'forward_model_{i}'] = self.forward_models[i].state_dict()
            state[f'inverse_model_{i}'] = self.inverse_models[i].state_dict()
            state[f'fm_optimizer_{i}']  = self.fm_optimizers[i].state_dict()
            state[f'im_optimizer_{i}']  = self.im_optimizers[i].state_dict()
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


class PerAgentSGAT(PerAgentGAT):
    """
    Stochastic GAT — each agent has its own stochastic forward + inverse model.

    Mirrors the relationship between SharedGAT → SharedSGAT:
      - forward_models use StochasticForwardModel
      - train_forward_step uses Gaussian NLL
      - ground_action samples from the distribution
    """

    def __init__(self, obs_size: int, n_actions: int, n_agents: int,
                 lr: float = 1e-3, device: str = 'cuda'):
        super().__init__(obs_size, n_actions, n_agents, lr, device)

        # Replace all deterministic FMs with stochastic ones
        self.forward_models = nn.ModuleList([
            StochasticForwardModel(self.obs_dim, n_actions).to(device)
            for _ in range(n_agents)
        ])
        self.fm_optimizers = [
            optim.Adam(fm.parameters(), lr=lr) for fm in self.forward_models
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
            # Sample from predicted distribution
            sampled_next = fm.sample(obs_t, a_onehot)
            logits       = im(obs_t, sampled_next)
            return logits.argmax(dim=-1).item()

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
        mean, log_std = fm(obs_t, a_onehot)
        loss = gaussian_nll_loss(mean, log_std, next_obs_t)
        opt.zero_grad(); loss.backward(); opt.step()
        return loss.item()

    def save(self, path: str):
        state = {}
        for i in range(self.n_agents):
            state[f'forward_model_{i}'] = self.forward_models[i].state_dict()
            state[f'inverse_model_{i}'] = self.inverse_models[i].state_dict()
            state[f'fm_optimizer_{i}']  = self.fm_optimizers[i].state_dict()
            state[f'im_optimizer_{i}']  = self.im_optimizers[i].state_dict()
        torch.save(state, path)
        print(f"[PerAgentSGAT] Saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        for i in range(self.n_agents):
            self.forward_models[i].load_state_dict(ckpt[f'forward_model_{i}'])
            self.inverse_models[i].load_state_dict(ckpt[f'inverse_model_{i}'])
            self.fm_optimizers[i].load_state_dict(ckpt[f'fm_optimizer_{i}'])
            self.im_optimizers[i].load_state_dict(ckpt[f'im_optimizer_{i}'])
        print(f"[PerAgentSGAT] Loaded from {path}")


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_gat(obs_size: int, n_actions: int, n_agents: int, variant: str,
              stochastic: bool, lr: float, device: str):
    """
    Instantiate the correct GAT class based on variant and stochastic flag.

    variant    stochastic    class
    ──────────────────────────────
    shared     False         SharedGAT
    shared     True          SharedSGAT
    per_agent  False         PerAgentGAT
    per_agent  True          PerAgentSGAT
    """
    if variant == 'shared':
        cls = SharedSGAT if stochastic else SharedGAT
    else:
        cls = PerAgentSGAT if stochastic else PerAgentGAT

    label = ('S' if stochastic else '') + 'GAT'
    print(f"[build_gat] Using {cls.__name__} ({label}, variant={variant})")
    return cls(obs_size=obs_size, n_actions=n_actions, n_agents=n_agents,
               lr=lr, device=device)


# ---------------------------------------------------------------------------
# Trajectory Collection
# ---------------------------------------------------------------------------

class TransitionBuffer:
    """Lightweight buffer storing (obs, action, next_obs) triples for GAT training."""

    def __init__(self):
        self._data = []

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
    obs_size:   int,
    policy:     IndependentDQN | None = None,
    epsilon:    float = 1.0,
    seed:       int   = 22,
    label:      str   = "env",
) -> list[TransitionBuffer]:
    """
    Collect (obs, action, next_obs) transitions from an environment.
    Returns one TransitionBuffer per agent.
    """
    env      = ExplorerMALocalObs(conf=env_config)
    n_agents = env.n_agents
    buffers  = [TransitionBuffer() for _ in range(n_agents)]

    print(f"\n[Collect] {label}: {n_episodes} episodes | "
          f"mode={env_config.get('env_mode')} | slip={env_config.get('slip_prob', 0.0)}")

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed)
        done   = False

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
            obs = next_obs

        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"  Episode {ep+1}/{n_episodes} | "
                  f"total transitions={sum(len(b) for b in buffers)}")

    env.close()
    return buffers


# ---------------------------------------------------------------------------
# GAT / SGAT Model Training
# ---------------------------------------------------------------------------

def train_gat_models(
    gat,
    real_buffers: list[TransitionBuffer],
    sim_buffers:  list[TransitionBuffer],
    n_epochs:     int  = 50,
    batch_size:   int  = 256,
    shared:       bool = True,
) -> dict:
    """
    Train forward models on real data, inverse models on sim data.

    GAT  — Forward: MSE loss          real (obs, action) → next_obs
    SGAT — Forward: Gaussian NLL loss  real (obs, action) → dist over next_obs
    Both — Inverse: cross-entropy      sim  (obs, next_obs) → action

    The training loop is identical regardless of deterministic vs stochastic;
    the loss difference is encapsulated inside gat.train_forward_step().
    """
    n_agents = len(real_buffers)
    history  = {'fm_losses': [], 'im_losses': []}

    is_sgat = isinstance(gat, (SharedSGAT, PerAgentSGAT))
    fm_desc = "Gaussian NLL (SGAT)" if is_sgat else "MSE (GAT)"

    print(f"\n{'='*60}")
    print(f"Training {'SGAT' if is_sgat else 'GAT'} models "
          f"({'shared' if shared else 'per-agent'})")
    print(f"  Forward model loss → {fm_desc}")
    print(f"  Forward data       → {sum(len(b) for b in real_buffers)} real transitions")
    print(f"  Inverse data       → {sum(len(b) for b in sim_buffers)} sim transitions")
    print(f"  Epochs: {n_epochs} | Batch: {batch_size}")
    print(f"{'='*60}")

    for epoch in range(n_epochs):
        epoch_fm, epoch_im = [], []

        if shared:
            real_obs, real_act, real_next = [], [], []
            sim_obs,  sim_act,  sim_next  = [], [], []
            for i in range(n_agents):
                ro, ra, rn = real_buffers[i].sample(batch_size // n_agents)
                so, sa, sn = sim_buffers[i].sample(batch_size // n_agents)
                real_obs.append(ro); real_act.append(ra); real_next.append(rn)
                sim_obs.append(so);  sim_act.append(sa);  sim_next.append(sn)

            fm_loss = gat.train_forward_step((
                np.concatenate(real_obs), np.concatenate(real_act), np.concatenate(real_next)
            ))
            im_loss = gat.train_inverse_step((
                np.concatenate(sim_obs), np.concatenate(sim_act), np.concatenate(sim_next)
            ))
            epoch_fm.append(fm_loss); epoch_im.append(im_loss)
        else:
            for i in range(n_agents):
                epoch_fm.append(gat.train_forward_step(i, real_buffers[i].sample(batch_size)))
                epoch_im.append(gat.train_inverse_step(i, sim_buffers[i].sample(batch_size)))

        history['fm_losses'].append(float(np.mean(epoch_fm)))
        history['im_losses'].append(float(np.mean(epoch_im)))

        if (epoch + 1) % max(1, n_epochs // 10) == 0:
            print(f"  Epoch {epoch+1:>4}/{n_epochs} | "
                  f"FM loss ({fm_desc}): {history['fm_losses'][-1]:.5f} | "
                  f"IM loss: {history['im_losses'][-1]:.5f}")

    print("GAT/SGAT model training complete.\n")
    return history


# ---------------------------------------------------------------------------
# Shared Evaluation Helper
# ---------------------------------------------------------------------------

def _eval_real(env_config, agent, obs_size, seed, device, n_episodes: int = 10):
    """
    Run n_episodes in the real (slip) environment with a greedy policy.
    Returns (mean_coverage, mean_reward) averaged over all episodes.
    Slip is stochastic so many episodes give a stable estimate.
    """
    all_coverages, all_rewards = [], []

    for _ in range(n_episodes):
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

        explored = np.count_nonzero(env.exploredMap)
        total    = env.SIZE[0] * env.SIZE[1]
        env.close()

        all_coverages.append(explored / total)
        all_rewards.append(float(np.mean(ep_reward)))

    return float(np.mean(all_coverages)), float(np.mean(all_rewards))


# ---------------------------------------------------------------------------
# MODE 1: Finetune a pre-trained checkpoint with GAT / SGAT
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
    finetune_epsilon: float = 0.05,
) -> IndependentDQN:
    """
    Fine-tune a pre-trained DQN in SIM using GAT/SGAT-grounded actions.

    Args:
        base_agent_path : Path to pre-trained IndependentDQN checkpoint
        gat             : Trained SharedGAT/SharedSGAT or PerAgentGAT/PerAgentSGAT
        env_config      : Sim environment config (for training rollouts)
        env_config_real : Real environment config (for evaluation)
        finetune_epsilon: Small exploration rate during finetuning (default 0.05)
    """
    is_sgat   = isinstance(gat, (SharedSGAT, PerAgentSGAT))
    algo_name = 'SGAT' if is_sgat else 'GAT'
    variant   = 'shared' if shared else 'per_agent'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir   = os.path.join(checkpoint_dir, f'{algo_name.lower()}_{variant}_finetune_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"FINETUNE MODE — {algo_name} variant: {variant}")
    print(f"  Base policy     : {base_agent_path}")
    print(f"  Episodes        : {n_episodes}")
    print(f"  Finetune epsilon: {finetune_epsilon}")
    print(f"  Eval env        : real slip_prob={env_config_real.get('slip_prob')}")
    print(f"  Best model by   : real-env COVERAGE (mean over {n_eval_episodes} eps)")
    print(f"  Run dir         : {run_dir}")
    print(f"{'='*60}\n")

    # Load base policy with small epsilon for stability
    agent = IndependentDQN(obs_size=obs_size, n_actions=4, device=device)
    agent.load(base_agent_path)
    agent.epsilon       = finetune_epsilon
    agent.epsilon_end   = finetune_epsilon
    agent.epsilon_decay = 1.0   # Hold epsilon constant during finetuning

    env      = ExplorerMALocalObs(conf=env_config)
    n_agents = env.n_agents

    ep_rewards, ep_coverages, losses          = [], [], []
    eval_coverages_hist, eval_rewards_hist    = [], []
    eval_ep_nums                              = []
    best_eval_coverage = -float('inf')

    for episode in range(n_episodes):
        obs, _    = env.reset(seed=seed)
        ep_reward = [0.0] * n_agents
        done      = False

        while not done:
            intended = [agent.select_action(obs[i]) for i in range(n_agents)]

            if shared:
                grounded = [gat.ground_action(obs[i], intended[i]) for i in range(n_agents)]
            else:
                grounded = [gat.ground_action(i, obs[i], intended[i]) for i in range(n_agents)]

            next_obs, rewards, terminated, truncated, _ = env.step(grounded)
            done = terminated or truncated

            for i in range(n_agents):
                agent.store_transition(obs[i], grounded[i], rewards[i], next_obs[i], done)
                ep_reward[i] += rewards[i]

            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            obs = next_obs

        coverage = np.count_nonzero(env.exploredMap) / (env.SIZE[0] * env.SIZE[1])
        ep_rewards.append(float(np.mean(ep_reward)))
        ep_coverages.append(coverage)

        if (episode + 1) % eval_freq == 0:
            eval_cov, eval_rew = _eval_real(
                env_config_real, agent, obs_size, seed, device, n_eval_episodes
            )
            eval_coverages_hist.append(eval_cov)
            eval_rewards_hist.append(eval_rew)
            eval_ep_nums.append(episode + 1)

            print(f"\n[EVAL/REAL] Episode {episode+1} | "
                  f"Coverage={eval_cov:.2%} | Reward={eval_rew:.2f} "
                  f"(mean over {n_eval_episodes} eps)")

            if eval_cov > best_eval_coverage:
                best_eval_coverage = eval_cov
                agent.save(os.path.join(run_dir, 'best_model.pt'))
                print(f"  *** New best coverage: {eval_cov:.2%} — model saved ***")

        if (episode + 1) % log_freq == 0:
            print(f"EP {episode+1:>5}/{n_episodes} | "
                  f"SimReward={np.mean(ep_rewards[-log_freq:]):.2f} | "
                  f"SimCov={np.mean(ep_coverages[-log_freq:]):.2%} | "
                  f"Loss={np.mean(losses[-200:]) if losses else 0:.4f} | "
                  f"ε={agent.epsilon:.3f}")

        if (episode + 1) % save_freq == 0:
            agent.save(os.path.join(run_dir, f'checkpoint_ep{episode+1}.pt'))

    agent.save(os.path.join(run_dir, 'final_model.pt'))
    np.save(os.path.join(run_dir, 'metrics.npy'), {
        'ep_rewards': ep_rewards, 'ep_coverages': ep_coverages, 'losses': losses,
        'eval_coverages': eval_coverages_hist, 'eval_rewards': eval_rewards_hist,
        'eval_episodes': eval_ep_nums, 'best_eval_coverage': best_eval_coverage,
        'mode': 'finetune', 'variant': variant, 'algo': algo_name,
        'slip_prob': env_config_real.get('slip_prob'),
    })

    print(f"\nFinetuning done. Best real-env coverage: {best_eval_coverage:.2%}")
    return agent


# ---------------------------------------------------------------------------
# MODE 2: Train IDQN from scratch inside the GAT / SGAT loop
# ---------------------------------------------------------------------------

def train_from_scratch_with_gat(
    gat,
    env_config:      dict,
    env_config_real: dict,
    obs_size:        int   = 7,
    n_episodes:      int   = 2000,
    shared:          bool  = True,
    save_freq:       int   = 200,
    log_freq:        int   = 20,
    eval_freq:       int   = 20,
    n_eval_episodes: int   = 20,
    checkpoint_dir:  str   = 'checkpoints',
    device:          str   = 'cuda',
    seed:            int   = 22,
    learning_rate:   float = 1e-4,
    epsilon_start:   float = 1.0,
    epsilon_end:     float = 0.05,
    epsilon_decay:   float = 0.995,
    buffer_capacity: int   = 100_000,
    batch_size:      int   = 128,
    target_update:   int   = 1000,
    gamma:           float = 0.99,
) -> IndependentDQN:
    """
    Train an IDQN policy FROM SCRATCH using GAT/SGAT-grounded actions throughout.

    For SGAT, the grounding is stochastic per call — the same intended action may
    map to different grounded actions across steps, naturally allowing the
    policy to handle stochastic transitions.

    Args:
        gat             : Trained SharedGAT/SGAT or PerAgentGAT/SGAT instance
        env_config      : Sim environment config (training rollouts)
        env_config_real : Real environment config (evaluation with slip)
        n_episodes      : Training episodes (recommend 1000-3000)
        n_eval_episodes : Real-env eval episodes to average (recommend 20-30)
    """
    is_sgat   = isinstance(gat, (SharedSGAT, PerAgentSGAT))
    algo_name = 'SGAT' if is_sgat else 'GAT'
    variant   = 'shared' if shared else 'per_agent'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir   = os.path.join(checkpoint_dir, f'{algo_name.lower()}_{variant}_scratch_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"FROM-SCRATCH MODE — {algo_name} variant: {variant}")
    print(f"  Episodes        : {n_episodes}")
    print(f"  Epsilon         : {epsilon_start} → {epsilon_end} (decay={epsilon_decay})")
    print(f"  Eval env        : real slip_prob={env_config_real.get('slip_prob')}")
    print(f"  Best model by   : real-env COVERAGE (mean over {n_eval_episodes} eps)")
    print(f"  Run dir         : {run_dir}")
    print(f"{'='*60}\n")

    env      = ExplorerMALocalObs(conf=env_config)
    n_agents = env.n_agents

    agent = IndependentDQN(
        obs_size=obs_size,
        n_actions=4,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update,
        device=device,
    )

    ep_rewards, ep_coverages, losses          = [], [], []
    eval_coverages_hist, eval_rewards_hist    = [], []
    eval_ep_nums                              = []
    best_eval_coverage = -float('inf')

    print(f"Training fresh IDQN with {algo_name} grounding ({n_agents} agents)...")
    print("-" * 60)

    for episode in range(n_episodes):
        obs, _    = env.reset(seed=seed)
        ep_reward = [0.0] * n_agents
        done      = False

        while not done:
            intended = [agent.select_action(obs[i]) for i in range(n_agents)]

            if shared:
                grounded = [gat.ground_action(obs[i], intended[i]) for i in range(n_agents)]
            else:
                grounded = [gat.ground_action(i, obs[i], intended[i]) for i in range(n_agents)]

            next_obs, rewards, terminated, truncated, _ = env.step(grounded)
            done = terminated or truncated

            for i in range(n_agents):
                agent.store_transition(obs[i], grounded[i], rewards[i], next_obs[i], done)
                ep_reward[i] += rewards[i]

            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            obs = next_obs

        agent.update_epsilon()

        coverage = np.count_nonzero(env.exploredMap) / (env.SIZE[0] * env.SIZE[1])
        ep_rewards.append(float(np.mean(ep_reward)))
        ep_coverages.append(coverage)

        if (episode + 1) % eval_freq == 0:
            eval_cov, eval_rew = _eval_real(
                env_config_real, agent, obs_size, seed, device, n_eval_episodes
            )
            eval_coverages_hist.append(eval_cov)
            eval_rewards_hist.append(eval_rew)
            eval_ep_nums.append(episode + 1)

            print(f"\n[EVAL/REAL] Episode {episode+1} | "
                  f"Coverage={eval_cov:.2%} | Reward={eval_rew:.2f} "
                  f"(mean over {n_eval_episodes} eps)")

            if eval_cov > best_eval_coverage:
                best_eval_coverage = eval_cov
                agent.save(os.path.join(run_dir, 'best_model.pt'))
                print(f"  *** New best coverage: {eval_cov:.2%} — model saved ***")

        if (episode + 1) % log_freq == 0:
            print(f"EP {episode+1:>5}/{n_episodes} | "
                  f"SimReward={np.mean(ep_rewards[-log_freq:]):.2f} | "
                  f"SimCov={np.mean(ep_coverages[-log_freq:]):.2%} | "
                  f"Loss={np.mean(losses[-200:]) if losses else 0:.4f} | "
                  f"ε={agent.epsilon:.3f} | "
                  f"Buffer={len(agent.replay_buffer)}")

        if (episode + 1) % save_freq == 0:
            agent.save(os.path.join(run_dir, f'checkpoint_ep{episode+1}.pt'))

    agent.save(os.path.join(run_dir, 'final_model.pt'))
    np.save(os.path.join(run_dir, 'metrics.npy'), {
        'ep_rewards': ep_rewards, 'ep_coverages': ep_coverages, 'losses': losses,
        'eval_coverages': eval_coverages_hist, 'eval_rewards': eval_rewards_hist,
        'eval_episodes': eval_ep_nums, 'best_eval_coverage': best_eval_coverage,
        'mode': 'scratch', 'variant': variant, 'algo': algo_name,
        'slip_prob': env_config_real.get('slip_prob'),
    })

    print(f"\nFrom-scratch training done. Best real-env coverage: {best_eval_coverage:.2%}")
    return agent


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def run_gat_pipeline(
    env_config_sim:        dict,
    env_config_real:       dict,
    variant:               str   = 'shared',
    mode:                  str   = 'scratch',
    stochastic:            bool  = False,
    base_agent_path:       str   = None,
    obs_size:              int   = 7,
    n_agents:              int   = 2,
    n_actions:             int   = 4,
    collect_episodes_real: int   = 200,
    collect_episodes_sim:  int   = 200,
    gat_epochs:            int   = 100,
    gat_batch_size:        int   = 256,
    gat_lr:                float = 1e-3,
    train_episodes:        int   = 2000,
    n_eval_episodes:       int   = 20,
    device:                str   = 'cuda',
    checkpoint_dir:        str   = 'checkpoints',
    seed:                  int   = 22,
    save_freq:             int   = 200,
    log_freq:              int   = 20,
    eval_freq:             int   = 20,
    finetune_epsilon:      float = 0.05,
):
    """
    End-to-end MA-GAT / MA-SGAT pipeline.

    stochastic=False → deterministic GAT (original)
    stochastic=True  → SGAT: forward model is a Gaussian, ground_action samples
                       from it, capturing real-world transition stochasticity.

    Steps:
      1. [scratch] No base policy needed  /  [finetune] Load base policy
      2. Collect real trajectories        → train forward model
      3. Collect sim trajectories         → train inverse model
      4. Build and train GAT/SGAT models
      5. Train/finetune IDQN with grounded actions, eval against real env
    """
    assert variant in ('shared', 'per_agent'), "variant must be 'shared' or 'per_agent'"
    assert mode in ('scratch', 'finetune'),    "mode must be 'scratch' or 'finetune'"
    if mode == 'finetune':
        assert base_agent_path is not None, "finetune mode requires --base_model"

    algo_name = 'SGAT' if stochastic else 'GAT'
    shared    = (variant == 'shared')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_root  = os.path.join(checkpoint_dir,
                             f'{algo_name.lower()}_{variant}_{mode}_{timestamp}')
    os.makedirs(run_root, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  MA-{algo_name} PIPELINE  |  variant={variant}  |  mode={mode}")
    print("=" * 70)

    # ── Step 1 ─────────────────────────────────────────────────────────────
    if mode == 'finetune':
        print("\n[Step 1] Loading base policy for trajectory collection ...")
        collection_policy = IndependentDQN(obs_size=obs_size, n_actions=n_actions, device=device)
        collection_policy.load(base_agent_path)
        collection_epsilon = 0.3
    else:
        print("\n[Step 1] Scratch mode — collecting with random policy ...")
        collection_policy  = None
        collection_epsilon = 1.0

    # ── Step 2: Real trajectories (forward model training data) ────────────
    print("\n[Step 2] Collecting real-world trajectories ...")
    real_buffers = collect_trajectories(
        env_config=env_config_real,
        n_episodes=collect_episodes_real,
        obs_size=obs_size,
        policy=collection_policy,
        epsilon=collection_epsilon,
        seed=seed,
        label="REAL",
    )
    print(f"  Real buffer sizes: {[len(b) for b in real_buffers]}")

    # ── Step 3: Sim trajectories (inverse model training data) ─────────────
    print("\n[Step 3] Collecting simulation trajectories ...")
    sim_buffers = collect_trajectories(
        env_config=env_config_sim,
        n_episodes=collect_episodes_sim,
        obs_size=obs_size,
        policy=collection_policy,
        epsilon=collection_epsilon,
        seed=seed,
        label="SIM",
    )
    print(f"  Sim buffer sizes: {[len(b) for b in sim_buffers]}")

    # ── Step 4: Build and train GAT/SGAT models ────────────────────────────
    print(f"\n[Step 4] Building and training {algo_name} models ...")
    gat = build_gat(
        obs_size=obs_size,
        n_actions=n_actions,
        n_agents=n_agents,
        variant=variant,
        stochastic=stochastic,
        lr=gat_lr,
        device=device,
    )

    gat_history = train_gat_models(
        gat=gat,
        real_buffers=real_buffers,
        sim_buffers=sim_buffers,
        n_epochs=gat_epochs,
        batch_size=gat_batch_size,
        shared=shared,
    )
    gat.save(os.path.join(run_root, f'{algo_name.lower()}_models.pt'))
    with open(os.path.join(run_root, f'{algo_name.lower()}_training_history.json'), 'w') as f:
        json.dump(gat_history, f, indent=2)

    # ── Step 5: Train/finetune IDQN with GAT/SGAT ─────────────────────────
    print(f"\n[Step 5] {'Finetuning' if mode == 'finetune' else 'Training from scratch'} "
          f"IDQN with {algo_name} ...")

    if mode == 'finetune':
        agent = finetune_with_gat(
            base_agent_path=base_agent_path,
            gat=gat,
            env_config=env_config_sim,
            env_config_real=env_config_real,
            obs_size=obs_size,
            n_episodes=train_episodes,
            shared=shared,
            save_freq=save_freq,
            log_freq=log_freq,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            checkpoint_dir=run_root,
            device=device,
            seed=seed,
            finetune_epsilon=finetune_epsilon,
        )
    else:
        agent = train_from_scratch_with_gat(
            gat=gat,
            env_config=env_config_sim,
            env_config_real=env_config_real,
            obs_size=obs_size,
            n_episodes=train_episodes,
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
    print(f"  MA-{algo_name} PIPELINE COMPLETE  |  variant={variant}  |  mode={mode}")
    print(f"  All outputs saved to: {run_root}")
    print("=" * 70)

    return agent, gat


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-Agent Grounded Action Transformation (MA-GAT / MA-SGAT)'
    )

    # ── Mode and variant ───────────────────────────────────────────────────
    parser.add_argument('--mode', type=str, default='scratch',
                        choices=['scratch', 'finetune'],
                        help='scratch: train IDQN from scratch (recommended)\n'
                             'finetune: load checkpoint and finetune')
    parser.add_argument('--variant', type=str, default='shared',
                        choices=['shared', 'per_agent'],
                        help='shared: parameter sharing across agents\n'
                             'per_agent: independent models per agent')
    parser.add_argument('--sgat', action='store_true', default=False,
                        help='Use Stochastic GAT (SGAT): the forward model outputs '
                             'a Gaussian distribution and samples from it during '
                             'grounding, capturing real-world stochasticity. '
                             'Recommended when slip_prob > 0.1.')
    parser.add_argument('--base_model', type=str, default=None,
                        help='Path to pre-trained IDQN checkpoint (required for finetune mode)')

    # ── Environment ────────────────────────────────────────────────────────
    parser.add_argument('--slip_prob', type=float, default=0.2,
                        help='Slip probability for real environment (default: 0.2)')
    parser.add_argument('--seed', type=int, default=22,
                        help='Map seed (default: 22)')

    # ── Data collection ────────────────────────────────────────────────────
    parser.add_argument('--collect_real', type=int, default=200,
                        help='Real-world collection episodes (default: 200)')
    parser.add_argument('--collect_sim', type=int, default=200,
                        help='Simulation collection episodes (default: 200)')

    # ── GAT/SGAT training ─────────────────────────────────────────────────
    parser.add_argument('--gat_epochs', type=int, default=100,
                        help='GAT/SGAT training epochs (default: 100)')
    parser.add_argument('--gat_lr', type=float, default=1e-3,
                        help='GAT/SGAT learning rate (default: 1e-3)')

    # ── IDQN training ─────────────────────────────────────────────────────
    parser.add_argument('--train_episodes', type=int, default=2000,
                        help='IDQN training episodes (default: 2000)')
    parser.add_argument('--n_eval_episodes', type=int, default=20,
                        help='Real-env eval episodes to average (default: 20)')
    parser.add_argument('--finetune_epsilon', type=float, default=0.05,
                        help='Epsilon for finetune mode (default: 0.05)')

    # ── Logistics ─────────────────────────────────────────────────────────
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--save_freq', type=int, default=200)
    parser.add_argument('--log_freq', type=int, default=20)
    parser.add_argument('--eval_freq', type=int, default=20)

    args = parser.parse_args()

    if args.mode == 'finetune' and args.base_model is None:
        parser.error("--mode finetune requires --base_model <path>")

    # ── Environment configuration ──────────────────────────────────────────
    conf["n_agents"]             = 2
    conf["shared_map"]           = True
    conf["size"]                 = [15, 15]
    conf["obstacles"]            = 0
    conf["lidar_range"]          = 2
    conf["obstacle_size"]        = [1, 3]
    conf["initial"]              = [1, 1]
    conf["collision_reward"]     = -50
    conf["out_of_bounds_reward"] = -50
    conf["movementCost"]         = 0.1
    conf["max_steps"]            = 200
    conf["verbose_slip"]         = False

    env_config_sim              = conf.copy()
    env_config_sim['env_mode']  = 'sim'
    env_config_sim['slip_prob'] = 0.0

    env_config_real              = conf.copy()
    env_config_real['env_mode']  = 'real'
    env_config_real['slip_prob'] = args.slip_prob

    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    algo   = 'SGAT' if args.sgat else 'GAT'
    print(f"Using device: {device}  |  Algorithm: {algo}")

    run_gat_pipeline(
        env_config_sim=env_config_sim,
        env_config_real=env_config_real,
        variant=args.variant,
        mode=args.mode,
        stochastic=args.sgat,
        base_agent_path=args.base_model,
        obs_size=7,
        n_agents=conf["n_agents"],
        n_actions=4,
        collect_episodes_real=args.collect_real,
        collect_episodes_sim=args.collect_sim,
        gat_epochs=args.gat_epochs,
        gat_lr=args.gat_lr,
        train_episodes=args.train_episodes,
        n_eval_episodes=args.n_eval_episodes,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        finetune_epsilon=args.finetune_epsilon,
    )