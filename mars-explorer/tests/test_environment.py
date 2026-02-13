import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import os
import glob
import re

# Import the environment
from mars_explorer.envs.explorer import ExplorerMALocalObs
from train_idqn import DQN_CNN, IndependentDQN

# Import the default config
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf


def visualize_local_obs(obs, agent_id, ax):
    """
    Visualize a 7x7 local observation grid.
    """
    ax.clear()
    ax.imshow(obs, cmap='RdYlGn_r', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(f'Agent {agent_id} Local Observation (7x7)')
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    
    # Add gridlines
    ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels
    for i in range(7):
        for j in range(7):
            value = obs[i, j]
            color = 'white' if value > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                   color=color, fontsize=8, weight='bold')
    
    # Add legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc='red', label='1.0 = Obstacle/Wall'),
        Rectangle((0, 0), 1, 1, fc='yellow', label='0.66 = Other Agent'),
        Rectangle((0, 0), 1, 1, fc='lightgreen', label='0.33 = Explored'),
        Rectangle((0, 0), 1, 1, fc='green', label='0.0 = Unexplored')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))


def visualize_global_map(env, ax):
    """
    Visualize the global map with agent positions.
    """
    ax.clear()
    
    # Create visualization map
    vis_map = np.zeros((env.sizeX, env.sizeY, 3))
    
    # Unexplored: dark gray
    vis_map[:, :] = [0.2, 0.2, 0.2]
    
    # Obstacles: black
    for obs_pos in env.obstacles_idx:
        vis_map[obs_pos[0], obs_pos[1]] = [0, 0, 0]
    
    # Explored: white
    explored_idx = np.where(env.exploredMap > 0)
    vis_map[explored_idx[0], explored_idx[1]] = [0.9, 0.9, 0.9]
    
    # Agent positions: different colors
    colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0]]  # Red, Blue, Green, Yellow
    for i, pos in enumerate(env.positions):
        color = colors[i % len(colors)]
        vis_map[pos[0], pos[1]] = color
        
        # Draw 7x7 observation window
        obs_radius = 3
        x_start = max(0, pos[0] - obs_radius)
        x_end = min(env.sizeX, pos[0] + obs_radius + 1)
        y_start = max(0, pos[1] - obs_radius)
        y_end = min(env.sizeY, pos[1] + obs_radius + 1)
        
        # Draw border around observation window
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                if x == x_start or x == x_end - 1 or y == y_start or y == y_end - 1:
                    if not (x == pos[0] and y == pos[1]):  # Don't overwrite agent position
                        vis_map[x, y] = [c * 0.5 for c in color]  # Darker border
    
    ax.imshow(vis_map, origin='upper')
    ax.set_title('Global Map View')
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    
    # Calculate and show coverage
    total_cells = env.SIZE[0] * env.SIZE[1]
    explored_cells = np.count_nonzero(env.exploredMap)
    coverage = explored_cells / total_cells
    ax.text(0.02, 0.98, f'Coverage: {coverage:.1%}', 
           transform=ax.transAxes, va='top', 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def test_environment_manual(env_config, n_steps=50, render_pygame=False):
    """
    Manually test the environment with random actions.
    Shows observations and rewards at each step.
    """
    env = ExplorerMALocalObs(conf=env_config)
    n_agents = env.n_agents
    
    # Create figure
    fig = plt.figure(figsize=(15, 5 * n_agents))
    
    # Create subplots: global map + local obs for each agent
    axes = []
    axes.append(fig.add_subplot(n_agents, 3, 1))  # Global map
    for i in range(n_agents):
        axes.append(fig.add_subplot(n_agents, 3, i * 3 + 2))  # Agent i local obs
    
    plt.ion()
    plt.show()
    
    print("=" * 80)
    print("TESTING ENVIRONMENT WITH RANDOM ACTIONS")
    print("=" * 80)
    
    obs, info = env.reset(seed=42)
    print(f"\nInitialized {n_agents} agents")
    print(f"Map size: {env.SIZE}")
    print(f"Observation shape per agent: {obs[0].shape}")
    print("\nStarting test...")
    print("-" * 80)
    
    for step in range(n_steps):
        print(f"\n{'='*80}")
        print(f"STEP {step + 1}/{n_steps}")
        print(f"{'='*80}")

        # Render with pygame if requested
        if render_pygame:
            try:
                env.render()
            except:
                print("Pygame rendering not available")

        # Visualize
        visualize_global_map(env, axes[0])
        for i in range(n_agents):
            visualize_local_obs(obs[i], i, axes[i + 1])
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.5)

        test = input("Press Enter to continue...")
        
        # Show current observations
        print("\nCurrent Observations:")
        for i in range(n_agents):
            print(f"\nAgent {i} at position {env.positions[i]}:")
            print(f"Observation shape: {obs[i].shape}")
            print(f"Observation values - Min: {obs[i].min():.2f}, Max: {obs[i].max():.2f}")
            print(f"Unique values: {np.unique(obs[i])}")
            print("7x7 Grid:")
            print(obs[i])
            
            # Count each type
            obstacles = np.sum(obs[i] == 1.0)
            unexplored = np.sum(obs[i] == 0.0)
            explored = np.sum(obs[i] == 0.33)
            other_agents = np.sum(obs[i] == 0.66)
            print(f"  Obstacles/Walls: {obstacles}")
            print(f"  Unexplored: {unexplored}")
            print(f"  Explored: {explored}")
            print(f"  Other agents: {other_agents}")
        
        # Random actions
        actions = [np.random.randint(0, 4) for _ in range(n_agents)]
        action_names = ['right', 'left', 'down', 'up']
        print(f"\nActions: {[f'Agent {i}: {action_names[actions[i]]}' for i in range(n_agents)]}")
        
        # Step
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        
        print(f"\nRewards: {rewards}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        print(f"New positions: {env.positions}")
        
        # Render with pygame if requested
        if render_pygame:
            try:
                env.render()
            except:
                print("Pygame rendering not available")
        
        # Visualize
        visualize_global_map(env, axes[0])
        for i in range(n_agents):
            visualize_local_obs(obs[i], i, axes[i + 1])
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.5)
        
        obs = next_obs
        
        if terminated or truncated:
            print(f"\n{'='*80}")
            print("EPISODE ENDED")
            print(f"Reason: {'Terminated' if terminated else 'Truncated'}")
            print(f"Total steps: {step + 1}")
            
            # Final statistics
            total_cells = env.SIZE[0] * env.SIZE[1]
            explored_cells = np.count_nonzero(env.exploredMap)
            coverage = explored_cells / total_cells
            print(f"Final coverage: {coverage:.2%}")
            print(f"Explored cells: {explored_cells}/{total_cells}")
            print(f"{'='*80}")
            break
    
    plt.ioff()
    plt.show()
    print("\nTest completed. Close the plot window to exit.")


def test_with_trained_model(env_config, model_path, n_episodes=5, max_steps=50, deterministic=True):
    """
    Test the environment with a trained model.
    
    Args:
        env_config: Environment configuration
        model_path: Path to model checkpoint
        n_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        deterministic: Use deterministic policy
    """
    env = ExplorerMALocalObs(conf=env_config)
    n_agents = env.n_agents
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load trained agent with correct observation size
    agent = IndependentDQN(obs_size=7, device=device)
    agent.load(model_path)
    agent.policy_net.eval()
    
    print("=" * 80)
    print("TESTING WITH TRAINED MODEL")
    print("=" * 80)
    print(f"Model loaded from: {model_path}")
    print(f"Device: {device}")
    print(f"Deterministic: {deterministic}")
    print(f"Number of test episodes: {n_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print("-" * 80)
    
    episode_rewards = []
    episode_lengths = []
    coverage_rates = []
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=42)
        episode_reward = [0] * n_agents
        episode_length = 0
        done = False
        
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        while not done and episode_length < max_steps:
            # Select actions deterministically
            actions = [agent.select_action(obs[i], eval_mode=True) for i in range(n_agents)]
            
            # Step
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            for i in range(n_agents):
                episode_reward[i] += rewards[i]
            
            obs = next_obs
            episode_length += 1
            
            # Optional: render
            try:
                env.render()
                time.sleep(0.05)
            except:
                pass
        
        # Calculate coverage
        total_cells = env.SIZE[0] * env.SIZE[1]
        explored_cells = np.count_nonzero(env.exploredMap)
        coverage = explored_cells / total_cells
        
        avg_reward = np.mean(episode_reward)
        episode_rewards.append(avg_reward)
        episode_lengths.append(episode_length)
        coverage_rates.append(coverage)
        
        print(f"  Total Reward: {avg_reward:.2f}")
        print(f"  Episode Length: {episode_length}")
        print(f"  Coverage: {coverage:.2%}")
        print(f"  Success: {terminated and coverage >= env.conf.get('explore_threshold', 0.9)}")
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average Coverage: {np.mean(coverage_rates):.2%} ± {np.std(coverage_rates):.2%}")
    print("=" * 80)


def get_checkpoint_episode_number(checkpoint_path):
    """Extract episode number from checkpoint filename."""
    match = re.search(r'checkpoint_ep(\d+)', os.path.basename(checkpoint_path))
    if match:
        return int(match.group(1))
    return 0


def find_all_checkpoints(run_dir):
    """Find all checkpoint files in a run directory, sorted by episode number."""
    checkpoint_pattern = os.path.join(run_dir, 'checkpoint_ep*.pt')
    checkpoints = glob.glob(checkpoint_pattern)
    
    # Sort by episode number
    checkpoints.sort(key=get_checkpoint_episode_number)
    
    return checkpoints


def test_all_checkpoints_sequential(env_config, run_dir, n_episodes=3, max_steps=50):
    """
    Test all checkpoints from a training run sequentially.
    
    Args:
        env_config: Environment configuration
        run_dir: Directory containing checkpoint files
        n_episodes: Number of episodes to test per checkpoint
        max_steps: Maximum steps per episode
    """
    # Find all checkpoints
    checkpoints = find_all_checkpoints(run_dir)
    
    if not checkpoints:
        print(f"No checkpoints found in {run_dir}")
        return
    
    print("=" * 80)
    print("TESTING ALL CHECKPOINTS SEQUENTIALLY")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Episodes per checkpoint: {n_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print("=" * 80)
    
    # Store results for all checkpoints
    all_results = []
    
    for checkpoint_path in checkpoints:
        episode_num = get_checkpoint_episode_number(checkpoint_path)
        print(f"\n{'='*80}")
        print(f"CHECKPOINT: Episode {episode_num}")
        print(f"Path: {checkpoint_path}")
        print(f"{'='*80}")
        
        env = ExplorerMALocalObs(conf=env_config)
        n_agents = env.n_agents
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load checkpoint
        agent = IndependentDQN(obs_size=7, device=device)
        agent.load(checkpoint_path)
        agent.policy_net.eval()
        
        # Test this checkpoint
        episode_rewards = []
        episode_lengths = []
        coverage_rates = []
        success_count = 0
        
        for episode in range(n_episodes):
            obs, info = env.reset(seed=42 + episode)  # Different seed for variety
            episode_reward = [0] * n_agents
            episode_length = 0
            done = False
            
            while not done and episode_length < max_steps:
                # Select actions deterministically
                actions = [agent.select_action(obs[i], eval_mode=True) for i in range(n_agents)]
                
                # Step
                next_obs, rewards, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                
                for i in range(n_agents):
                    episode_reward[i] += rewards[i]
                
                obs = next_obs
                episode_length += 1
                
                # Optional: render (can be slow for many checkpoints)
                # try:
                #     env.render()
                #     time.sleep(0.01)
                # except:
                #     pass
            
            # Calculate coverage
            total_cells = env.SIZE[0] * env.SIZE[1]
            explored_cells = np.count_nonzero(env.exploredMap)
            coverage = explored_cells / total_cells
            
            # Check success
            is_success = terminated and coverage >= env_config.get('explore_threshold', 0.9)
            if is_success:
                success_count += 1
            
            avg_reward = np.mean(episode_reward)
            episode_rewards.append(avg_reward)
            episode_lengths.append(episode_length)
            coverage_rates.append(coverage)
        
        # Summary for this checkpoint
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        avg_coverage = np.mean(coverage_rates)
        success_rate = success_count / n_episodes
        
        print(f"\nCheckpoint Episode {episode_num} Results:")
        print(f"  Avg Reward: {avg_reward:.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  Avg Length: {avg_length:.1f} ± {np.std(episode_lengths):.1f}")
        print(f"  Avg Coverage: {avg_coverage:.2%} ± {np.std(coverage_rates):.2%}")
        print(f"  Success Rate: {success_rate:.1%} ({success_count}/{n_episodes})")
        
        all_results.append({
            'episode': episode_num,
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'avg_coverage': avg_coverage,
            'success_rate': success_rate
        })
    
    # Final summary across all checkpoints
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS ALL CHECKPOINTS")
    print("=" * 80)
    print(f"{'Episode':<10} {'Avg Reward':<15} {'Avg Length':<15} {'Coverage':<15} {'Success Rate':<15}")
    print("-" * 80)
    for result in all_results:
        print(f"{result['episode']:<10} {result['avg_reward']:<15.2f} {result['avg_length']:<15.1f} "
              f"{result['avg_coverage']:<15.2%} {result['success_rate']:<15.1%}")
    print("=" * 80)
    
    # Plot progress over checkpoints
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Progress Across Checkpoints', fontsize=14, fontweight='bold')
        
        episodes = [r['episode'] for r in all_results]
        
        # Plot rewards
        axes[0, 0].plot(episodes, [r['avg_reward'] for r in all_results], 'o-', linewidth=2)
        axes[0, 0].set_xlabel('Training Episode')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].set_title('Reward Progress')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot episode length
        axes[0, 1].plot(episodes, [r['avg_length'] for r in all_results], 'o-', linewidth=2, color='orange')
        axes[0, 1].set_xlabel('Training Episode')
        axes[0, 1].set_ylabel('Average Episode Length')
        axes[0, 1].set_title('Episode Length Progress')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot coverage
        axes[1, 0].plot(episodes, [r['avg_coverage'] for r in all_results], 'o-', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Training Episode')
        axes[1, 0].set_ylabel('Average Coverage')
        axes[1, 0].set_title('Coverage Progress')
        axes[1, 0].axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Target (90%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot success rate
        axes[1, 1].plot(episodes, [r['success_rate'] for r in all_results], 'o-', linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Training Episode')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_title('Success Rate Progress')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(run_dir, 'checkpoint_evaluation.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")
        
        plt.show()
    except Exception as e:
        print(f"\nCould not create plot: {e}")


if __name__ == '__main__':
    # Environment configuration
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

    
    print("\nChoose test mode:")
    print("1. Manual test with random actions (visualize observations)")
    print("2. Test with trained model (single checkpoint)")
    print("3. Test all checkpoints sequentially (evaluate training progress)")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == '1':
        # Test with random actions
        render_pygame = input("Enable pygame rendering? (y/n): ").strip().lower() == 'y'
        n_steps = int(input("Number of steps to test (default 50): ") or "50")
        test_environment_manual(conf, n_steps=n_steps, render_pygame=render_pygame)
    
    elif choice == '2':
        # Test with trained model
        model_path = input("Enter path to trained model: ").strip()
        n_episodes = int(input("Number of episodes to test (default 5): ") or "5")
        max_steps = int(input("Max steps per episode (default 50): ") or "50")
        test_with_trained_model(conf, model_path, n_episodes=n_episodes, max_steps=max_steps)
    
    elif choice == '3':
        # Test all checkpoints sequentially
        run_dir = input("Enter path to training run directory (e.g., checkpoints/idqn_run_20260212_170426): ").strip()
        n_episodes = int(input("Number of episodes per checkpoint (default 3): ") or "3")
        max_steps = int(input("Max steps per episode (default 50): ") or "50")
        test_all_checkpoints_sequential(conf, run_dir, n_episodes=n_episodes, max_steps=max_steps)
    
    else:
        print("Invalid choice. Running manual test by default.")
        test_environment_manual(conf, n_steps=50, render_pygame=False)