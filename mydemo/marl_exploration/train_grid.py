"""
Training Script for Grid Exploration Task
==========================================
Train MARL agents to explore and cover a 2D grid map.

Reward Design:
- +10 for exploring a new cell
- -0.1 for revisiting a cell
- +100 bonus for 100% coverage
- Team reward based on coverage percentage

Usage:
    python train_grid.py
"""

import os
import sys
import argparse
import numpy as np
from copy import deepcopy

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Register the custom environment
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environments import REGISTRY

from xuance.common import load_yaml, recursive_dict_update
from xuance.environment import make_envs
from xuance.environment.multi_agent_env import REGISTRY_MULTI_AGENT_ENV
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import IQL_Agents


def parse_args():
    parser = argparse.ArgumentParser(description="Train Grid Exploration MARL")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--parallels", type=int, default=8)
    parser.add_argument("--running-steps", type=int, default=2000000)
    parser.add_argument("--eval-interval", type=int, default=50000)
    parser.add_argument("--test-episodes", type=int, default=5)
    parser.add_argument("--config", type=str,
                        default="configs/grid_exploration.yaml")
    parser.add_argument("--benchmark", action="store_true", default=False)
    return parser.parse_args()


def print_config(configs):
    """Print training configuration."""
    print("=" * 60)
    print("Grid Exploration - Multi-Agent RL Training")
    print("=" * 60)
    print(f"  Algorithm: {configs.agent}")
    print(f"  Environment: {configs.env_id}")
    print(f"  Grid Size: {configs.grid_size} x {configs.grid_size}")
    print(f"  Number of Agents: {configs.num_agents}")
    print(f"  Device: {configs.device}")
    print(f"  Parallels: {configs.parallels}")
    print(f"  Running Steps: {configs.running_steps}")
    print("=" * 60)


def main():
    parser = parse_args()

    # Load configuration
    config_path = parser.config
    configs_dict = load_yaml(file_dir=config_path)

    # Override with command line args
    cmd_dict = {k: v for k, v in parser.__dict__.items() if v is not None}
    configs_dict = recursive_dict_update(configs_dict, cmd_dict)
    configs = argparse.Namespace(**configs_dict)

    # Set seed
    set_seed(configs.seed)
    # Add env_seed for xuance (required by make_envs)
    configs.env_seed = configs.seed

    # Register custom environment with xuance
    REGISTRY_MULTI_AGENT_ENV.update(REGISTRY)

    # Create environments
    envs = make_envs(configs)

    # Create agent (IQL works well for discrete actions)
    Agent = IQL_Agents(config=configs, envs=envs)

    # Print config
    print_config(configs)

    if configs.benchmark:
        print("\n[Benchmark Mode] Train + Evaluate")
        print("-" * 40)

        # Create test environment
        configs_test = deepcopy(configs)
        configs_test.parallels = 1
        configs_test.render = True
        configs_test.render_mode = 'human'
        test_envs = make_envs(configs_test)

        # Training loop
        train_steps = configs.running_steps // configs.parallels
        eval_interval = configs.eval_interval // configs.parallels
        test_episode = configs.test_episodes
        num_updates = train_steps // eval_interval

        best_coverage = 0.0

        for epoch in range(num_updates):
            print(f"\nEpoch {epoch + 1}/{num_updates}")
            print(f"  Training steps: {epoch * eval_interval}/{train_steps}")

            # Train
            Agent.train(eval_interval)

            # Evaluate
            test_scores = Agent.test(
                test_episodes=test_episode,
                test_envs=test_envs,
                close_envs=False
            )

            mean_reward = np.mean(test_scores) if len(test_scores) > 0 else 0
            print(f"  Evaluation - Mean Reward: {mean_reward:.2f}")

            # Track best
            if mean_reward > best_coverage:
                best_coverage = mean_reward
                Agent.save_model(model_name="best_model.pth")
                print(f"  New best model saved! Coverage: {best_coverage:.2f}")

        print("\n" + "=" * 60)
        print("Benchmark Complete!")
        print(f"Best Coverage: {best_coverage:.2f}")
        print(f"Logs: {configs.log_dir}")
        print("=" * 60)

    else:
        print("\n[Training Mode]")
        Agent.train(configs.running_steps // configs.parallels)
        Agent.save_model("final_model.pth")
        print("Training complete!")

    Agent.finish()


if __name__ == "__main__":
    main()