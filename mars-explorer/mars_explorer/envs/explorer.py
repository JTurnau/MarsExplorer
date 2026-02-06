import numpy as np
import pygame as pg

from mars_explorer.utils.randomMapGenerator import Generator
from mars_explorer.utils.lidarSensor import Lidar
from mars_explorer.render.viewer import Viewer
from mars_explorer.envs.settings import DEFAULT_CONFIG

import gym
from gym import spaces

class ExplorerMA(gym.Env):
    metadata = {'render.modes': ['rgb_array'],
                'video.frames_per_second': 6}

    def __init__(self, conf=None):
        self.conf = DEFAULT_CONFIG if conf is None else conf

        self.sizeX, self.sizeY = self.conf["size"]
        self.SIZE = self.conf["size"]
        self.movementCost = self.conf["movementCost"]
        self.n_agents = self.conf.get("n_agents", 1)
        self.shared_map = self.conf.get("shared_map", True)

        self.last_actions = [0, 0]

        self.action_space = spaces.MultiDiscrete([4]*self.n_agents)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.sizeX, self.sizeY, 2),
            dtype=np.float32
        )


        self.viewerActive = False

    def reset(self, seed=None):
        self.maxSteps = self.conf["max_steps"]

        # generate map with a fixed seed
        gen = Generator(self.conf, seed=seed)
        randomMap = gen.get_map().astype(np.double)
        randomMapOriginal = randomMap.copy()
        randomMap[randomMap == 1.0] = 1.0
        randomMap[randomMap == 0.0] = 0.3
        self.groundTruthMap = randomMap

        # lidar for each agent
        self.ldrs = [Lidar(r=self.conf["lidar_range"],
                          channels=self.conf["lidar_channels"],
                          map=randomMapOriginal) for _ in range(self.n_agents)]

        # obstacles
        obstacles_idx = np.where(self.groundTruthMap == 1.0)
        self.obstacles_idx = [list(i) for i in np.stack((obstacles_idx[0], obstacles_idx[1]), axis=1)]

        # shared explored map
        self.exploredMap = np.zeros(self.SIZE, dtype=np.double)

        # initialize agent positions
        initial = self.conf.get("initial", [0,0])
        self.positions = []
        for i in range(self.n_agents):
            x = initial[0] + i
            y = initial[1]
            self.positions.append([x, y])

        # trajectories and rewards
        self.state_trajectory = [[] for _ in range(self.n_agents)]
        self.reward_trajectory = [[] for _ in range(self.n_agents)]
        self.drone_trajectory = [[] for _ in range(self.n_agents)]

        self.timeStep = 0
        self.dones = [False]*self.n_agents
        self.rewards = [0]*self.n_agents

        # activate lidars and update map
        for i in range(self.n_agents):
            self._activateLidar(i)
        self._updateMaps()

        return [self._get_obs(i) for i in range(self.n_agents)]

    def _choice(self, agent_idx, action):
        dx, dy = 0, 0

        print(f"Agent {agent_idx} choice")

        if action == 0: dx = 1
        elif action == 1: dx = -1
        elif action == 2: dy = 1
        elif action == 3: dy = -1

        # If the environment mode is set to 'real', apply uniform probability to slip
        if self.conf["env_mode"] == "real":
            if np.random.rand() < self.conf["slip_prob"]:
                print(f"Agent {agent_idx} slipped")
                dx, dy = 0, 0
        
        self._move(agent_idx, dx, dy)
            

    def _get_obs(self, agent_idx):
        obs = np.zeros((self.sizeX, self.sizeY, 3), dtype=np.float32)

        # Channel 0: shared explored map
        obs[:, :, 0] = self.exploredMap

        # Channel 1: this agent's position
        x, y = self.positions[agent_idx]
        obs[x, y, 1] = 1.0

        # Channel 2: other agents' positions
        for i, (ox, oy) in enumerate(self.positions):
            if i != agent_idx:
                obs[ox, oy, 2] = 1.0

        return obs


    def _move(self, agent_idx, dx, dy):
        candX = self.positions[agent_idx][0] + dx
        candY = self.positions[agent_idx][1] + dy

        in_bounds = 0 <= candX < self.sizeX and 0 <= candY < self.sizeY
        in_obstacle = [candX, candY] in self.obstacles_idx

        if in_bounds and not in_obstacle:
            self.positions[agent_idx] = [candX, candY]
        else:
            self.dones[agent_idx] = True
            if not in_bounds:
                self.rewards[agent_idx] = self.conf["out_of_bounds_reward"]
            elif in_obstacle:
                self.rewards[agent_idx] = self.conf["collision_reward"]

    def _activateLidar(self, agent_idx):
        self.ldrs[agent_idx].update(self.positions[agent_idx])
        self.lidarIndexes = getattr(self, 'lidarIndexes', {})
        self.lidarIndexes[agent_idx] = self.ldrs[agent_idx].idx

    def _updateMaps(self):
        # shared explored map
        self.pastExploredMap = self.exploredMap.copy()
        
        # accumulate lidar readings from all agents
        for idx in range(self.n_agents):
            lidarX = self.lidarIndexes[idx][:,0]
            lidarY = self.lidarIndexes[idx][:,1]
            self.exploredMap[lidarX, lidarY] = self.groundTruthMap[lidarX, lidarY]

        # mark agent positions as explored; removed in favor of channels
        # for pos in self.positions:
        #     self.exploredMap[pos[0], pos[1]] = 0.6

    def _computeReward(self):
        new_explored = int(np.count_nonzero(self.exploredMap))
        old_explored = int(np.count_nonzero(self.pastExploredMap))
        reward_increment = new_explored - old_explored
        for i in range(self.n_agents):
            if self.rewards[i] == 0:
                self.rewards[i] = float(reward_increment - self.movementCost)

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()  # convert np array to list

        self.rewards = [0]*self.n_agents

        self.last_actions = actions

        for i, action in enumerate(actions):
            if not self.dones[i]:
                self._choice(i, int(action))   # move agent
                self._activateLidar(i)


        self._updateMaps()
        self._computeReward()
        self.timeStep += 1

        # ------------------- Check for agent-agent collisions -------------------
        positions_seen = {}
        for i, pos in enumerate(self.positions):
            pos_tuple = tuple(pos)
            if pos_tuple in positions_seen:
                # collision: mark both agents as done
                collided_agent = positions_seen[pos_tuple]
                self.dones[i] = True
                self.dones[collided_agent] = True
                self.rewards[i] = self.conf.get("collision_reward", -400)
                self.rewards[collided_agent] = self.conf.get("collision_reward", -400)
            else:
                positions_seen[pos_tuple] = i
        # ------------------------------------------------------------------------

        # check done due to max steps or full exploration
        if self.timeStep >= self.maxSteps or np.count_nonzero(self.exploredMap) > 0.95*(self.SIZE[0]*self.SIZE[1]):
            self.dones = [True]*self.n_agents

        # update trajectories
        for i in range(self.n_agents):
            self.state_trajectory[i].append(np.reshape(self.exploredMap, (self.sizeX, self.sizeY,1)))
            self.reward_trajectory[i].append(self.rewards[i])
            self.drone_trajectory[i].append(self.positions[i].copy())

        obs = [self._get_obs(i) for i in range(self.n_agents)]
        
        return obs, self.rewards, self.dones, {}



    def render(self, mode='human'):
        if not hasattr(self, "viewer") or self.viewer is None:
            self.viewer = Viewer(self, self.conf["viewer"])
            self.viewerActive = True
        try:
            self.viewer.run()
            return np.swapaxes(self.viewer.get_display_as_array(), 0, 1)
        except pg.error:
            # if the window was closed, recreate the viewer
            self.viewer = Viewer(self, self.conf["viewer"])
            self.viewerActive = True
            self.viewer.run()
            return np.swapaxes(self.viewer.get_display_as_array(), 0, 1)

    def close(self):
        if self.viewerActive:
            self.viewer.quit()
