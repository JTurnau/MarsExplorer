import gym
import numpy as np
import math
import time
import argparse
from mars_explorer.envs.explorer import ExplorerMA
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf

import mars_explorer

def get_conf():
    conf["size"] = [30, 30]
    conf["obstacles"] = 20
    conf["lidar_range"] = 4
    conf["obstacle_size"] = [1,3]

    conf["viewer"]["night_color"] = (0, 0, 0)
    conf["viewer"]["draw_lidar"] = True

    # conf["viewer"]["width"] = conf["size"][0]*42
    # conf["viewer"]["width"] = conf["size"][1]*42

    conf["viewer"]["drone_img"] = "../img/drone.png"
    conf["viewer"]["obstacle_img"] = "../img/block.png"
    conf["viewer"]["background_img"] = "../img/mars.jpg"
    conf["viewer"]["light_mask"] = "../img/light_350_hard.png"
    return conf

def getArgs():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '-w', '--warm-up',
        default=0,
        type=int,
        help='Number of warm up games ')
    argparser.add_argument(
        '-g', '--games',
        default=10,
        type=int,
        help='Games to be played')
    argparser.add_argument(
        '-s', '--save',
        default=False,
        action="store_true",
        help='Save each rendered image')
    return argparser.parse_args()

if __name__ == "__main__":
    # Multi-agent config
    conf["n_agents"] = 2
    conf["shared_map"] = True
    conf["size"] = [30, 30]
    conf["obstacles"] = 20
    conf["lidar_range"] = 4
    conf["obstacle_size"] = [1,3]
    conf["env_mode"] = "sim"
    conf["slip_prob"] = 1.0

    seed = 42

    env = ExplorerMA(conf=conf)

observations = env.reset(seed=seed)
done = [False] * env.n_agents
episode_reward = 0.0

for step in range(20):
    print(f"Step {step}")
    print(f"Agent positions: {env.positions}")
    print(f"Explored cells: {np.count_nonzero(env.exploredMap)}")
    print(f"Dones: {done}")

    env.render()

    # Sample one action for each agent
    actions = env.action_space.sample()
    observations, rewards, done, info = env.step(actions)

    # get obs for each agent
    obs_list = [env._get_obs(i) for i in range(env.n_agents)]

    # for agent_idx, obs in enumerate(obs_list):
    #     print(f"\nAgent {agent_idx} observation shape: {obs.shape}")
    #     for channel in range(obs.shape[2]):
    #         print(f"--- Channel {channel} ---")
    #         print(obs[:, :, channel])

    for i in range(env.n_agents):
        if done[i] and rewards[i] != -400:
            rewards[i] = 0

    print(f"Actions taken: {actions}")
    print(f"Rewards: {rewards}")
    print(f"Dones after step: {done}")

    episode_reward += sum(rewards)

    if all(done):
        print("Episode finished")
        print("Episode reward:", episode_reward)
        episode_reward = 0.0
        observation = env.reset(seed=seed)
        done = [False]*env.n_agents

    test = input("hi")
    #time.sleep(0.3)
    print("-"*40)

    env.close()

