import numpy as np
import pygame as pg
import random

from mars_explorer.utils.randomMapGenerator import Generator
from mars_explorer.utils.lidarSensor import Lidar
from mars_explorer.render.viewer import Viewer
from mars_explorer.envs.settings import DEFAULT_CONFIG

import gymnasium as gym
from gymnasium import spaces


class ExplorerMALocalObs(gym.Env):
    """
    Multi-agent Mars Explorer with 7x7 local observations.
    Each agent observes a 7x7 grid around itself with:
    - 1.0 for obstacles/walls
    - 0.0 for unexplored squares
    - 0.33 for explored squares
    - 0.66 for other agents
    
    Episode terminates if ANY agent collides or goes out of bounds.
    Uses Gymnasium API (returns 5 values from step).
    
    MODIFICATION FOR SIM-TO-REAL TRANSFER:
    - When env_mode="sim": deterministic transitions (no slip)
    - When env_mode="real": stochastic transitions with slip_prob
      * With probability (1 - slip_prob): execute intended action
      * With probability slip_prob: execute uniformly random action
    """
    metadata = {'render.modes': ['rgb_array'],
                'video.frames_per_second': 6}

    def __init__(self, conf=None):
        super().__init__()
        self.conf = DEFAULT_CONFIG if conf is None else conf

        self.sizeX, self.sizeY = self.conf["size"]
        self.SIZE = self.conf["size"]
        self.movementCost = self.conf["movementCost"]
        self.n_agents = self.conf.get("n_agents", 1)
        self.shared_map = self.conf.get("shared_map", True)

        self.last_actions = [0] * self.n_agents

        # Local observation window size
        self.obs_size = 7
        self.obs_radius = self.obs_size // 2  # 2 cells in each direction

        # Action space: 4 discrete actions (right, left, up, down)
        self.action_space = spaces.Discrete(4)

        # Observation space: 7x7 grid with single channel
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_size, self.obs_size),
            dtype=np.float32
        )

        # Shared explored map
        self.exploredMap = np.zeros(self.SIZE, dtype=np.double)
        self.viewerActive = False

    def seed(self, seed=None):
        """Set random seed."""
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)

        self.np_random = np.random.RandomState(seed)
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        """Reset environment. Returns (observation, info) tuple."""
        # CRITICAL: Only use seed for map generation, not for episode RNG
        # This ensures the map is fixed but slip events are random across episodes
        
        self.maxSteps = self.conf["max_steps"]
        self.exploration_done = False

        # Generate map with the provided seed (for consistent map layout)
        gen = Generator(self.conf, seed=seed)
        randomMap = gen.get_map().astype(np.double)
        randomMapOriginal = randomMap.copy()
        randomMap[randomMap == 1.0] = 1.0
        randomMap[randomMap == 0.0] = 0.3
        self.groundTruthMap = randomMap

        # Lidar for each agent
        self.ldrs = [Lidar(r=self.conf["lidar_range"],
                          channels=self.conf["lidar_channels"],
                          map=randomMapOriginal) for _ in range(self.n_agents)]

        # Obstacles
        obstacles_idx = np.where(self.groundTruthMap == 1.0)
        self.obstacles_idx = [list(i) for i in np.stack((obstacles_idx[0], obstacles_idx[1]), axis=1)]

        # Shared explored map
        self.exploredMap = np.zeros(self.SIZE, dtype=np.double)

        # Initialize agent positions
        initial = self.conf.get("initial")
        self.positions = []
        first_placement = True
        for i in range(self.n_agents):
            if first_placement:
                x, y = initial
            else:
                x, y = initial[0] + 2, initial[1]
            first_placement = False
            self.positions.append([x, y])

        # Trajectories and rewards
        self.state_trajectory = [[] for _ in range(self.n_agents)]
        self.reward_trajectory = [[] for _ in range(self.n_agents)]
        self.drone_trajectory = [[] for _ in range(self.n_agents)]

        self.timeStep = 0
        self.dones = [False] * self.n_agents
        self.rewards = [0] * self.n_agents

        # Activate lidars and update map
        for i in range(self.n_agents):
            self._activateLidar(i)
        self._updateMaps()

        # Gymnasium returns (obs, info) tuple
        obs = [self._get_local_obs(i) for i in range(self.n_agents)]
        info = {}

        return obs, info

    def _get_local_obs(self, agent_idx):
        """
        Get 7x7 local observation around the agent.
        Values:
        - 1.0 for obstacles and walls (out of bounds)
        - 0.0 for unexplored squares
        - 0.33 for explored squares
        - 0.66 for other agents
        """
        obs = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
        
        agent_x, agent_y = self.positions[agent_idx]
        
        for i in range(self.obs_size):
            for j in range(self.obs_size):
                # Calculate global coordinates
                global_x = agent_x + (i - self.obs_radius)
                global_y = agent_y + (j - self.obs_radius)
                
                # Check if out of bounds (walls)
                if global_x < 0 or global_x >= self.sizeX or global_y < 0 or global_y >= self.sizeY:
                    obs[i, j] = 1.0  # Wall
                    continue
                
                # Check if obstacle
                if [global_x, global_y] in self.obstacles_idx:
                    obs[i, j] = 1.0  # Obstacle
                    continue
                
                # Check if another agent is at this position
                is_other_agent = False
                for other_idx, (ox, oy) in enumerate(self.positions):
                    if other_idx != agent_idx and ox == global_x and oy == global_y:
                        obs[i, j] = 0.66  # Other agent
                        is_other_agent = True
                        break
                
                if is_other_agent:
                    continue
                
                # Check if explored
                if self.exploredMap[global_x, global_y] != 0:
                    obs[i, j] = 0.33  # Explored
                else:
                    obs[i, j] = 0.0  # Unexplored
        
        return obs

    def _choice(self, agent_idx, action):
        """
        Execute action with potential slip based on environment mode.
        
        SIM-TO-REAL TRANSFER MECHANISM:
        - env_mode="sim": Execute intended action deterministically
        - env_mode="real": With probability slip_prob, execute random action instead
        
        IMPORTANT: Uses np.random.default_rng() for slip to ensure true randomness
        across episodes even when map seed is fixed.
        """
        actual_action = action
        
        # Apply slip probability if in real mode
        if self.conf.get("env_mode") == "real":
            slip_rng = np.random.default_rng()
            if slip_rng.random() < self.conf.get("slip_prob", 0.0):
                # Slip: take a uniformly random action instead of intended action
                other_actions = [a for a in range(4) if a != action]
                actual_action = int(slip_rng.choice(other_actions))
                if self.conf.get("verbose_slip", False):
                    action_names = ['right', 'left', 'down', 'up']
                    print(f"Agent {agent_idx} slipped! Intended: {action_names[action]}, Actual: {action_names[actual_action]}")
        
        dx, dy = 0, 0
        if actual_action == 0: dx = 1
        elif actual_action == 1: dx = -1
        elif actual_action == 2: dy = 1
        elif actual_action == 3: dy = -1
        
        self._move(agent_idx, dx, dy)

    def _move(self, agent_idx, dx, dy):
        candX = self.positions[agent_idx][0] + dx
        candY = self.positions[agent_idx][1] + dy

        in_bounds = 0 <= candX < self.sizeX and 0 <= candY < self.sizeY
        in_obstacle = [candX, candY] in self.obstacles_idx

        if in_bounds and not in_obstacle:
            self.positions[agent_idx] = [candX, candY]
        else:
            # Mark agent as collided/out of bounds
            self.dones[agent_idx] = True
            if not in_bounds:
                self.rewards[agent_idx] = self.conf.get("out_of_bounds_reward", -50)
            elif in_obstacle:
                self.rewards[agent_idx] = self.conf.get("collision_reward", -50)

    def _activateLidar(self, agent_idx):
        self.ldrs[agent_idx].update(self.positions[agent_idx])
        self.lidarIndexes = getattr(self, 'lidarIndexes', {})
        self.lidarIndexes[agent_idx] = self.ldrs[agent_idx].idx

    def _updateMaps(self):
        # Shared explored map
        self.pastExploredMap = self.exploredMap.copy()
        
        # Accumulate lidar readings from all agents
        for idx in range(self.n_agents):
            lidarX = self.lidarIndexes[idx][:, 0]
            lidarY = self.lidarIndexes[idx][:, 1]
            self.exploredMap[lidarX, lidarY] = self.groundTruthMap[lidarX, lidarY]

    def _computeReward(self):
        """Compute individual rewards for each agent based on exploration."""
        claimed = np.zeros(self.SIZE, dtype=bool)
        for i in range(self.n_agents):
            # If agent already has a collision/out-of-bounds penalty, don't override it
            if self.rewards[i] != 0:
                continue

            lidar_idx = self.lidarIndexes[i]
            new_cells = 0
            for x, y in lidar_idx:
                if self.exploredMap[x, y] != self.pastExploredMap[x, y] and not claimed[x, y]:
                    new_cells += 1
                    claimed[x, y] = True  # Mark cell as claimed

            # Individual reward = new cells explored - movement cost
            self.rewards[i] = float(new_cells - self.movementCost)

    def step(self, actions):
        """
        Step the environment. Returns (obs, reward, terminated, truncated, info).
        
        Episode terminates if ANY agent collides or goes out of bounds.
        Each agent receives individual rewards based on their own exploration.
        """
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()

        self.rewards = [0] * self.n_agents
        self.last_actions = actions

        # Execute actions for each agent
        for i, action in enumerate(actions):
            if not self.dones[i]:
                self._choice(i, int(action))
                self._activateLidar(i)

        self._updateMaps()
        self._computeReward()
        self.timeStep += 1

        # Check for agent-agent collisions
        positions_seen = {}
        for i, pos in enumerate(self.positions):
            pos_tuple = tuple(pos)
            if pos_tuple in positions_seen:
                # Collision between agents - both get penalty
                collided_agent = positions_seen[pos_tuple]
                self.dones[i] = True
                self.dones[collided_agent] = True
                self.rewards[i] = self.conf.get("collision_reward", -50)
                self.rewards[collided_agent] = self.conf.get("collision_reward", -50)
            else:
                positions_seen[pos_tuple] = i

        # Check termination conditions
        terminated = False
        truncated = False

        # Episode terminates if ANY agent crashed or went out of bounds
        if any(self.dones):
            terminated = True
            # Apply penalty to failing agent when episode terminates due to collision
            for i in range(self.n_agents):
                if self.dones[i] and self.rewards[i] < 0:
                    # This agent caused termination - keep their penalty
                    pass

        # Exploration-based success termination
        total_cells = self.SIZE[0] * self.SIZE[1]
        explored_cells = np.count_nonzero(self.exploredMap)
        coverage = explored_cells / total_cells

        if (not self.exploration_done) and coverage >= self.conf.get("explore_threshold", 0.90):
            self.exploration_done = True
            # Bonus reward for completing exploration
            for i in range(self.n_agents):
                self.rewards[i] += self.conf.get("explore_bonus", 500)
            terminated = True

        # Max steps termination (truncation, not termination)
        if self.timeStep >= self.maxSteps:
            truncated = True

        # Update done flags
        if terminated or truncated:
            self.dones = [True] * self.n_agents

        # Update trajectories
        for i in range(self.n_agents):
            self.state_trajectory[i].append(self.exploredMap.copy())
            self.reward_trajectory[i].append(self.rewards[i])
            self.drone_trajectory[i].append(self.positions[i].copy())

        obs = [self._get_local_obs(i) for i in range(self.n_agents)]
        info = {}

        # Gymnasium API: return 5 values
        return obs, self.rewards, terminated, truncated, info

    def render(self, mode='human'):
        if not hasattr(self, "viewer") or self.viewer is None:
            self.viewer = Viewer(self, self.conf["viewer"])
            self.viewerActive = True
        try:
            self.viewer.run()
            return np.swapaxes(self.viewer.get_display_as_array(), 0, 1)
        except pg.error:
            self.viewer = Viewer(self, self.conf["viewer"])
            self.viewerActive = True
            self.viewer.run()
            return np.swapaxes(self.viewer.get_display_as_array(), 0, 1)

    def close(self):
        if self.viewerActive:
            self.viewer.quit()