"""
Multi-Agent RL for Robot Swarm Exploration
==========================================
This example demonstrates how to use XuanCe framework
for multi-agent reinforcement learning in robot swarm exploration tasks.

Algorithms: MADDPG (Multi-Agent DDPG)
Environment: MPE (Multi-Agent Particle Environment) - simple_spread scenario
"""

import argparse
import numpy as np
from copy import deepcopy
from xuance.common import load_yaml, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import MADDPG_Agents


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Agent RL for Robot Swarm Exploration using MADDPG"
    )
    parser.add_argument("--env-id", type=str, default="simple_spread_v3",
                        help="MPE scenario: simple_spread_v3 (N agents, N landmarks)")
    parser.add_argument("--algo", type=str, default="maddpg",
                        help="Algorithm: maddpg, mappo, iql, etc.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Training device: cuda:0 or cpu")
    parser.add_argument("--parallels", type=int, default=16,
                        help="Number of parallel environments")
    parser.add_argument("--running-steps", type=int, default=5000000,
                        help="Total training steps")
    parser.add_argument("--test", action="store_true",
                        help="Run test mode with trained model")
    parser.add_argument("--benchmark", action="store_true", default=True,
                        help="Run benchmark mode (train + eval)")
    parser.add_argument("--config", type=str,
                        default="configs/maddpg_exploration.yaml",
                        help="Path to config file")

    return parser.parse_args()


def print_train_info(configs):
    """Print training configuration."""
    print("=" * 60)
    print("Multi-Agent RL for Robot Swarm Exploration")
    print("=" * 60)
    info = {
        "Algorithm": configs.agent,
        "Environment": f"{configs.env_name} - {configs.env_id}",
        "Device": configs.device,
        "Parallel Envs": configs.parallels,
        "Training Steps": configs.running_steps,
        "Actor LR": configs.learning_rate_actor,
        "Critic LR": configs.learning_rate_critic,
        "Gamma": configs.gamma,
    }
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    # Parse arguments
    parser = parse_args()

    # Load configuration
    config_dir = parser.config.rsplit("/", 1)[0] + "/"
    configs_dict = load_yaml(file_dir=parser.config)
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    # Set random seed for reproducibility
    set_seed(configs.seed)

    # Create vectorized environments
    # MPE simple_spread: N agents must cover N landmarks
    envs = make_envs(configs)

    # Create MARL agents
    Agent = MADDPG_Agents(config=configs, envs=envs)

    # Print training info
    print_train_info(configs)

    if configs.benchmark:
        print("\n[Mode] Benchmark: Train + Evaluate")
        print("-" * 40)

        # Create test environments
        configs_test = deepcopy(configs)
        configs_test.parallels = configs_test.test_episode
        test_envs = make_envs(configs_test)

        # Training loop
        train_steps = configs.running_steps // configs.parallels
        eval_interval = configs.eval_interval // configs.parallels
        test_episode = configs.test_episode
        num_epoch = int(train_steps / eval_interval)

        best_score = -np.inf
        scores_history = []

        for epoch in range(num_epoch):
            print(f"\nEpoch {epoch + 1}/{num_epoch}")
            print(f"  Steps: {epoch * eval_interval}/{train_steps}")

            # Train
            Agent.train(eval_interval)

            # Evaluate
            test_scores = Agent.test(
                test_episodes=test_episode,
                test_envs=test_envs,
                close_envs=False
            )
            mean_score = np.mean(test_scores)
            std_score = np.std(test_scores)

            print(f"  Evaluation - Mean: {mean_score:.2f}, Std: {std_score:.2f}")

            scores_history.append({
                "epoch": epoch + 1,
                "mean": mean_score,
                "std": std_score,
                "step": Agent.current_step
            })

            # Save best model
            if mean_score > best_score:
                best_score = mean_score
                Agent.save_model(model_name="best_model.pth")
                print(f"  New best model saved! Score: {best_score:.2f}")

        print("\n" + "=" * 60)
        print("Benchmark Complete!")
        print(f"Best Score: {best_score:.2f}")
        print(f"Logs saved to: {configs.log_dir}")
        print(f"Models saved to: {configs.model_dir}")
        print("=" * 60)

    else:
        if configs.test:
            print("\n[Mode] Test with trained model")
            print("-" * 40)
            configs.parallels = configs.test_episode
            test_envs = make_envs(configs)
            Agent.load_model(path=Agent.model_dir_load)
            scores = Agent.test(
                test_episodes=configs.test_episode,
                test_envs=test_envs,
                close_envs=True
            )
            print(f"Test Results - Mean: {np.mean(scores):.2f}, Std: {np.std(scores):.2f}")

        else:
            print("\n[Mode] Training only")
            print("-" * 40)
            Agent.train(configs.running_steps // configs.parallels)
            Agent.save_model("final_train_model.pth")
            print("Training complete!")

    Agent.finish()
    print("\nDone!")