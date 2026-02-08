import time
import torch
import numpy as np
import os

from mars_explorer.envs.explorer import ExplorerMA
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf
from mappo_rover import MAPPOAgent

# -----------------------------
# USER CONFIG
# -----------------------------
RUN_NAME = "MAPPO_MarsExplorer__1770491175"
CHECKPOINT_FILE = "mappo_update_0350.pt"

CHECKPOINT_PATH = os.path.join(
    "checkpoints",
    RUN_NAME,
    CHECKPOINT_FILE
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load checkpoint
# -----------------------------
ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

print(f"Loaded checkpoint from update {ckpt['update']}")

conf = ckpt.get("conf", conf)

conf["initial"] = [1,1]

n_agents = conf["n_agents"]

# -----------------------------
# Load env
# -----------------------------
env = ExplorerMA(conf=conf)
obs_list = env.reset(seed=42)

obs_shape = env._get_obs(0).shape
action_space = env.action_space

# -----------------------------
# Load trained policy
# -----------------------------
agent = MAPPOAgent(obs_shape, action_space, n_agents).to(device)
agent.load_state_dict(ckpt["model_state_dict"])
agent.eval()

# -----------------------------
# Rollout
# -----------------------------
episode_reward = 0.0
step = 0

env.render()

while True:
    obs = torch.tensor(
        np.array(obs_list),
        dtype=torch.float32
    ).permute(0, 3, 1, 2).to(device)

    # Deterministic MAPPO actions
    with torch.no_grad():
        actions = agent.act_deterministic(obs)
        print(f"Actions taken: {actions}")

    obs_list, rewards, dones, info = env.step(actions)

    episode_reward += sum(rewards)

    env.render()
    time.sleep(0.5)

    step += 1

    if any(dones):
        print("=" * 50)
        print(f"Episode finished in {step} steps")
        print(f"Total team reward: {episode_reward:.2f}")
        print("=" * 50)
        break
