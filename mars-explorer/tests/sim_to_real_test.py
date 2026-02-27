"""
Sim-to-Real Transfer Testing Script

This script evaluates a DQN model trained in simulation on the "real" environment
with slip dynamics to assess sim-to-real transfer performance.

Usage:
    python test_sim_to_real.py --model_path <path_to_model> --slip_prob <probability>
"""

import numpy as np
import torch
import argparse
import os
from datetime import datetime
import json
from scipy import stats

# Import the environment
from mars_explorer.envs.explorer import ExplorerMALocalObs
from train_idqn import DQN_CNN, IndependentDQN

# Import the default config
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf


def evaluate_sim_to_real_transfer(
    model_path,
    env_config_sim,
    env_config_real,
    n_real_episodes=100,
    map_seed=22,
    max_steps=100,
    device='cuda',
    verbose=False,
    save_results=True,
    results_dir='sim_to_real_results'
):
    """
    Evaluate sim-to-real transfer by testing a sim-trained model in real environment.
    
    Strategy:
    - Sim evaluation: 1 deterministic episode on map_seed (baseline reference)
    - Real evaluation: n_real_episodes with same map but random slip events
    
    Args:
        model_path: Path to the trained model checkpoint
        env_config_sim: Environment configuration for simulation (used as reference)
        env_config_real: Environment configuration for real environment (with slip)
        n_real_episodes: Number of real evaluation episodes (with random slip)
        map_seed: Seed for map generation (same for all episodes)
        max_steps: Maximum steps per episode
        device: 'cuda' or 'cpu'
        verbose: Print detailed episode information
        save_results: Save results to file
        results_dir: Directory to save results
    
    Returns:
        Dictionary containing evaluation metrics
    """
    
    print("=" * 80)
    print("SIM-TO-REAL TRANSFER EVALUATION")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Map seed: {map_seed} (fixed for all episodes)")
    print(f"Simulation mode: {env_config_sim.get('env_mode', 'sim')}")
    print(f"Real mode: {env_config_real.get('env_mode', 'real')}")
    print(f"Slip probability: {env_config_real.get('slip_prob', 0.0)}")
    print(f"Sim episodes: 1 (deterministic baseline)")
    print(f"Real episodes: {n_real_episodes} (random slip events)")
    print(f"Device: {device}")
    print("=" * 80)
    
    # Load trained agent
    agent = IndependentDQN(obs_size=7, device=device)
    agent.load(model_path)
    agent.policy_net.eval()
    
    # Create environments
    env_sim = ExplorerMALocalObs(conf=env_config_sim)
    env_real = ExplorerMALocalObs(conf=env_config_real)
    n_agents = env_sim.n_agents
    
    # Metrics storage
    sim_results = {
        'episode_reward': 0,
        'episode_length': 0,
        'coverage_rate': 0,
        'success': False,
        'collision': False,
        'truncation': False
    }
    
    real_results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'coverage_rates': [],
        'success_count': 0,
        'collision_count': 0,
        'out_of_bounds_count': 0,
        'truncation_count': 0
    }
    
    # Evaluate in SIMULATION (single deterministic episode)
    print("\n" + "=" * 80)
    print("EVALUATING IN SIMULATION (DETERMINISTIC BASELINE)")
    print(f"Running 1 episode with map seed {map_seed}")
    print("=" * 80)
    
    obs, info = env_sim.reset(seed=map_seed)
    episode_reward = [0] * n_agents
    episode_length = 0
    done = False
    step_num = 0
    
    while not done and episode_length < max_steps:
        # Select actions deterministically
        actions = [agent.select_action(obs[i], eval_mode=True) for i in range(n_agents)]
        
        # Step environment
        next_obs, rewards, terminated, truncated, info = env_sim.step(actions)
        done = terminated or truncated
        
        for i in range(n_agents):
            episode_reward[i] += rewards[i]
        
        obs = next_obs
        episode_length += 1
        step_num += 1
    
    # Calculate coverage
    total_cells = env_sim.SIZE[0] * env_sim.SIZE[1]
    explored_cells = np.count_nonzero(env_sim.exploredMap)
    coverage = explored_cells / total_cells
    
    # Check termination reason
    is_success = terminated and coverage >= env_config_sim.get('explore_threshold', 0.90)
    is_collision = terminated and any(r < -10 for r in rewards)
    is_truncated = truncated
    
    # Store sim results
    avg_reward = np.mean(episode_reward)
    sim_results['episode_reward'] = avg_reward
    sim_results['episode_length'] = episode_length
    sim_results['coverage_rate'] = coverage
    sim_results['success'] = is_success
    sim_results['collision'] = is_collision
    sim_results['truncation'] = is_truncated
    
    print("\n" + "-" * 80)
    print("SIMULATION RESULTS (1 deterministic episode):")
    print(f"  Reward: {avg_reward:.2f}")
    print(f"  Length: {episode_length}")
    print(f"  Coverage: {coverage:.2%}")
    print(f"  Success: {is_success}")
    print(f"  Collision: {is_collision}")
    print(f"  Truncated: {is_truncated}")
    print("-" * 80)
    
    # Evaluate in REAL environment (multiple episodes with random slip)
    print("\n" + "=" * 80)
    print("EVALUATING IN REAL ENVIRONMENT (WITH RANDOM SLIP)")
    print(f"Running {n_real_episodes} episodes with map seed {map_seed}")
    print(f"Slip events are random (drawn independently each timestep)")
    print("=" * 80)
    
    for episode in range(n_real_episodes):
        # Reset with same map seed, but slip will be random due to step-by-step RNG
        obs, info = env_real.reset(seed=map_seed)
        episode_reward = [0] * n_agents
        episode_length = 0
        done = False
        step_num = 0
        
        while not done and episode_length < max_steps:
            # Select actions deterministically (same policy as sim)
            actions = [agent.select_action(obs[i], eval_mode=True) for i in range(n_agents)]
            
            # Step environment (slip may occur randomly here)
            next_obs, rewards, terminated, truncated, info = env_real.step(actions)
            done = terminated or truncated
            
            for i in range(n_agents):
                episode_reward[i] += rewards[i]
            
            obs = next_obs
            episode_length += 1
            step_num += 1
        
        # Calculate coverage
        total_cells = env_real.SIZE[0] * env_real.SIZE[1]
        explored_cells = np.count_nonzero(env_real.exploredMap)
        coverage = explored_cells / total_cells
        
        # Check termination reason
        is_success = terminated and coverage >= env_config_real.get('explore_threshold', 0.90)
        is_collision = terminated and any(r < -10 for r in rewards)
        is_truncated = truncated
        
        # Update metrics
        avg_reward = np.mean(episode_reward)
        real_results['episode_rewards'].append(avg_reward)
        real_results['episode_lengths'].append(episode_length)
        real_results['coverage_rates'].append(coverage)
        
        if is_success:
            real_results['success_count'] += 1
        if is_collision:
            real_results['collision_count'] += 1
        if is_truncated:
            real_results['truncation_count'] += 1
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"Real Episode {episode + 1}/{n_real_episodes}: "
                  f"Reward={avg_reward:.2f}, Length={episode_length}, Steps={step_num}, Coverage={coverage:.2%}")
    
    # Calculate real statistics
    real_stats = {
        'mean_reward': np.mean(real_results['episode_rewards']),
        'std_reward': np.std(real_results['episode_rewards'], ddof=1),
        'mean_length': np.mean(real_results['episode_lengths']),
        'std_length': np.std(real_results['episode_lengths'], ddof=1),
        'mean_coverage': np.mean(real_results['coverage_rates']),
        'std_coverage': np.std(real_results['coverage_rates'], ddof=1),
        'success_rate': real_results['success_count'] / n_real_episodes,
        'collision_rate': real_results['collision_count'] / n_real_episodes,
        'truncation_rate': real_results['truncation_count'] / n_real_episodes
    }

    n = len(real_results['episode_rewards'])

    # Coverage CI
    coverage_ci = stats.t.interval(
        0.95,
        df=n-1,
        loc=real_stats['mean_coverage'],
        scale=stats.sem(real_results['coverage_rates'])
    )

    # Reward CI
    reward_ci = stats.t.interval(
        0.95,
        df=n-1,
        loc=real_stats['mean_reward'],
        scale=stats.sem(real_results['episode_rewards'])
    )

    # Length CI
    length_ci = stats.t.interval(
        0.95,
        df=n-1,
        loc=real_stats['mean_length'],
        scale=stats.sem(real_results['episode_lengths'])
    )
    real_stats['length_ci_95'] = length_ci

    real_stats['coverage_ci_95'] = coverage_ci
    real_stats['reward_ci_95'] = reward_ci
    
    print("\n" + "-" * 80)
    print(f"REAL ENVIRONMENT RESULTS ({n_real_episodes} episodes with random slip):")
    print(f"  Mean Reward: {real_stats['mean_reward']:.2f} ± {real_stats['std_reward']:.2f}")
    print(f"  Reward 95% CI:  [{reward_ci[0]:.2f}, {reward_ci[1]:.2f}]")
    print(f"  Mean Length: {real_stats['mean_length']:.1f} ± {real_stats['std_length']:.1f}")
    print(f"  Length 95% CI:  [{length_ci[0]:.1f}, {length_ci[1]:.1f}]")
    print(f"  Mean Coverage: {real_stats['mean_coverage']:.2%} ± {real_stats['std_coverage']:.2%}")
    print(f"  Coverage 95% CI:[{coverage_ci[0]:.2%}, {coverage_ci[1]:.2%}]")
    print(f"  Success Rate: {real_stats['success_rate']:.1%}")
    print(f"  Collision Rate: {real_stats['collision_rate']:.1%}")
    print(f"  Truncation Rate: {real_stats['truncation_rate']:.1%}")
    print("-" * 80)
    
    # Calculate transfer gap (comparing sim baseline to real mean)
    print("\n" + "=" * 80)
    print("SIM-TO-REAL TRANSFER GAP ANALYSIS")
    print("=" * 80)
    
    reward_gap = sim_results['episode_reward'] - real_stats['mean_reward']
    coverage_gap = sim_results['coverage_rate'] - real_stats['mean_coverage']
    success_gap = (1.0 if sim_results['success'] else 0.0) - real_stats['success_rate']
    
    reward_gap_pct = (reward_gap / sim_results['episode_reward'] * 100) if sim_results['episode_reward'] != 0 else 0
    coverage_gap_pct = (coverage_gap / sim_results['coverage_rate'] * 100) if sim_results['coverage_rate'] != 0 else 0
    
    print(f"  Sim Reward (baseline): {sim_results['episode_reward']:.2f}")
    print(f"  Real Reward (mean):    {real_stats['mean_reward']:.2f} ± {real_stats['std_reward']:.2f}")
    print(f"  Reward Gap: {reward_gap:.2f} ({reward_gap_pct:.1f}% degradation)")
    print()
    print(f"  Sim Coverage (baseline): {sim_results['coverage_rate']:.2%}")
    print(f"  Real Coverage (mean):    {real_stats['mean_coverage']:.2%} ± {real_stats['std_coverage']:.2%}")
    print(f"  Coverage Gap: {coverage_gap:.2%} ({coverage_gap_pct:.1f}% degradation)")
    print()
    print(f"  Sim Success: {sim_results['success']}")
    print(f"  Real Success Rate: {real_stats['success_rate']:.1%}")
    print(f"  Success Gap: {success_gap:.2%}")
    print("=" * 80)
    
    # Prepare results dictionary
    results = {
        'model_path': model_path,
        'map_seed': map_seed,
        'slip_prob': env_config_real.get('slip_prob', 0.0),
        'n_sim_episodes': 1,
        'n_real_episodes': n_real_episodes,
        'simulation': {
            'single_episode': sim_results,
            'reward': sim_results['episode_reward'],
            'coverage': sim_results['coverage_rate'],
            'success': sim_results['success']
        },
        'real': {
            'raw_results': real_results,
            'statistics': real_stats
        },
        'transfer_gap': {
            'reward_gap': reward_gap,
            'reward_gap_pct': reward_gap_pct,
            'coverage_gap': coverage_gap,
            'coverage_gap_pct': coverage_gap_pct,
            'success_gap': success_gap
        }
    }
    
    # Save results
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        slip_str = f"slip{int(env_config_real.get('slip_prob', 0.0) * 100)}"
        results_filename = f'sim_to_real_{slip_str}_seed{map_seed}_{timestamp}.json'
        results_path = os.path.join(results_dir, results_filename)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_to_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
    
    return results


def compare_multiple_slip_probabilities(
    model_path,
    env_config_base,
    slip_probs=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    n_real_episodes=100,
    map_seed=22,
    max_steps=100,
    device='cuda',
    results_dir='sim_to_real_results'
):
    """
    Compare performance across multiple slip probabilities.
    
    Args:
        model_path: Path to trained model
        env_config_base: Base environment configuration
        slip_probs: List of slip probabilities to test
        n_real_episodes: Number of real episodes per slip probability
        map_seed: Seed for map generation (same map for all tests)
        max_steps: Maximum steps per episode
        device: 'cuda' or 'cpu'
        results_dir: Directory to save results
    """
    
    print("=" * 80)
    print("MULTI-SLIP PROBABILITY COMPARISON")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Map seed: {map_seed} (fixed for all tests)")
    print(f"Slip probabilities to test: {slip_probs}")
    print(f"Real episodes per slip prob: {n_real_episodes}")
    print("=" * 80)
    
    all_results = []
    
    for slip_prob in slip_probs:
        print(f"\n{'='*80}")
        print(f"Testing with slip_prob = {slip_prob}")
        print(f"{'='*80}")
        
        # Configure sim environment (no slip)
        env_config_sim = env_config_base.copy()
        env_config_sim['env_mode'] = 'sim'
        env_config_sim['slip_prob'] = 0.0
        
        # Configure real environment (with slip)
        env_config_real = env_config_base.copy()
        env_config_real['env_mode'] = 'real'
        env_config_real['slip_prob'] = slip_prob
        
        # Evaluate
        results = evaluate_sim_to_real_transfer(
            model_path=model_path,
            env_config_sim=env_config_sim,
            env_config_real=env_config_real,
            n_real_episodes=n_real_episodes,
            map_seed=map_seed,
            max_steps=max_steps,
            device=device,
            verbose=False,
            save_results=True,
            results_dir=results_dir
        )
        
        all_results.append(results)
    
    # Print summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS ALL SLIP PROBABILITIES")
    print("=" * 80)
    print(f"Map seed: {map_seed} (same map for all tests)")
    print(f"Sim baseline: 1 deterministic episode")
    print(f"Real per slip: {n_real_episodes} episodes with random slip events")
    print("-" * 80)
    print(f"{'Slip Prob':<12} {'Real Reward':<15} {'Real Coverage':<15} {'Success Rate':<15} {'Reward Gap %':<15}")
    print("-" * 80)
    
    for result in all_results:
        slip = result['slip_prob']
        real_reward = result['real']['statistics']['mean_reward']
        real_coverage = result['real']['statistics']['mean_coverage']
        success_rate = result['real']['statistics']['success_rate']
        gap_pct = result['transfer_gap']['reward_gap_pct']
        
        print(f"{slip:<12.2f} {real_reward:<15.2f} {real_coverage:<15.2%} "
              f"{success_rate:<15.2%} {gap_pct:<15.1f}%")
    
    print("=" * 80)
    
    # Save combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_path = os.path.join(results_dir, f'multi_slip_comparison_seed{map_seed}_{timestamp}.json')
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    all_results_serializable = convert_to_serializable(all_results)
    
    with open(combined_path, 'w') as f:
        json.dump(all_results_serializable, f, indent=2)
    
    print(f"\nCombined results saved to: {combined_path}")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sim-to-Real Transfer Evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--slip_prob', type=float, default=0.3,
                       help='Slip probability for real environment (default: 0.3)')
    parser.add_argument('--n_real_episodes', type=int, default=100,
                       help='Number of real episodes with random slip (default: 100)')
    parser.add_argument('--map_seed', type=int, default=22,
                       help='Seed for map generation (default: 22)')
    parser.add_argument('--max_steps', type=int, default=100,
                       help='Maximum steps per episode (default: 100)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--multi_slip', action='store_true',
                       help='Test multiple slip probabilities')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed episode information')
    parser.add_argument('--results_dir', type=str, default='sim_to_real_results',
                       help='Directory to save results (default: sim_to_real_results)')
    
    args = parser.parse_args()
    
    # Environment configuration
    conf["n_agents"] = 2
    conf["shared_map"] = True
    conf["size"] = [15, 15]
    conf["obstacles"] = 0
    conf["lidar_range"] = 2
    conf["obstacle_size"] = [1, 3]
    conf["initial"] = [1, 1]
    conf["collision_reward"] = -50
    conf["out_of_bounds_reward"] = -50
    conf["movementCost"] = 0.1
    conf["max_steps"] = args.max_steps
    conf["verbose_slip"] = False  # Set to True to see slip events
    
    if args.multi_slip:
        # Test multiple slip probabilities
        slip_probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        compare_multiple_slip_probabilities(
            model_path=args.model_path,
            env_config_base=conf,
            slip_probs=slip_probs,
            n_real_episodes=args.n_real_episodes,
            map_seed=args.map_seed,
            max_steps=args.max_steps,
            device=args.device,
            results_dir=args.results_dir
        )
    else:
        # Test single slip probability
        # Sim configuration
        env_config_sim = conf.copy()
        env_config_sim['env_mode'] = 'sim'
        env_config_sim['slip_prob'] = 0.0
        
        # Real configuration
        env_config_real = conf.copy()
        env_config_real['env_mode'] = 'real'
        env_config_real['slip_prob'] = args.slip_prob
        
        evaluate_sim_to_real_transfer(
            model_path=args.model_path,
            env_config_sim=env_config_sim,
            env_config_real=env_config_real,
            n_real_episodes=args.n_real_episodes,
            map_seed=args.map_seed,
            max_steps=args.max_steps,
            device=args.device,
            verbose=args.verbose,
            save_results=True,
            results_dir=args.results_dir
        )