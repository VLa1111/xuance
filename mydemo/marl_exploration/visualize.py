"""
Visualization for Robot Swarm Exploration
==========================================
This script provides visualization tools for:
1. TensorBoard logs
2. Render trained models
3. Plot training curves
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize MARL Training")
    parser.add_argument("--log-dir", type=str,
                        default="logs/maddpg_exploration/",
                        help="TensorBoard log directory")
    parser.add_argument("--model-dir", type=str,
                        default="models/maddpg_exploration/",
                        help="Model checkpoint directory")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to render")
    return parser.parse_args()


def plot_training_curves(log_dir, save_path=None):
    """Plot training curves from TensorBoard logs."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("Please install tensorboard: pip install tensorboard")
        return

    print(f"Loading logs from: {log_dir}")

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # Available scalars
    print("\nAvailable metrics:")
    for tag in ea.Tags()['scalars']:
        print(f"  - {tag}")

    # Plot key metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Training Curves - Multi-Agent RL for Swarm Exploration", fontsize=14)

    metrics_to_plot = [
        ("Test_Episode_Rewards/Mean", "Mean Episode Reward"),
        ("Test_Episode_Rewards/Std", "Std Episode Reward"),
        ("Training_Episode_Rewards/Mean", "Training Reward"),
        ("Loss_Actor/Mean", "Actor Loss"),
    ]

    for idx, (tag, title) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        try:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            ax.plot(steps, values)
            ax.set_title(title)
            ax.set_xlabel("Steps")
            ax.grid(True, alpha=0.3)
        except KeyError:
            ax.text(0.5, 0.5, f"No data for\n{tag}", ha='center', va='center')
            ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved to: {save_path}")
    else:
        plt.savefig("training_curves.png", dpi=150)
        print("Training curves saved to: training_curves.png")

    plt.show()


def render_trained_model(model_dir, env_id="simple_spread_v3", episodes=5):
    """Render episodes using a trained model."""
    print(f"\nRendering {episodes} episodes from: {model_dir}")

    from xuance.common import load_yaml
    from xuance.environment import make_envs
    from xuance.torch.agents import MADDPG_Agents
    import argparse

    # Load config
    config_path = Path(__file__).parent / "configs" / "maddpg_exploration.yaml"
    configs_dict = load_yaml(file_dir=str(config_path))
    configs = argparse.Namespace(**configs_dict)

    # Override for rendering
    configs.env_id = env_id
    configs.render = True
    configs.render_mode = 'human'
    configs.parallels = 1
    configs.device = "cpu"

    # Create environment
    envs = make_envs(configs)

    # Create agent and load model
    Agent = MADDPG_Agents(config=configs, envs=envs)

    model_path = Path(model_dir) / "best_model.pth"
    if model_path.exists():
        Agent.load_model(path=str(model_path))
        print(f"Loaded model from: {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")
        return

    # Run episodes
    print(f"\nRendering {episodes} episodes (close window to continue)...")
    scores = Agent.test(test_episodes=episodes, test_envs=envs, close_envs=True)

    print(f"\nResults over {episodes} episodes:")
    print(f"  Mean Score: {np.mean(scores):.2f}")
    print(f"  Std Score: {np.std(scores):.2f}")
    print(f"  Min Score: {np.min(scores):.2f}")
    print(f"  Max Score: {np.max(scores):.2f}")

    Agent.finish()


def check_logs(log_dir):
    """List available log files."""
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Log directory not found: {log_dir}")
        return

    print(f"\nLogs in {log_dir}:")
    for f in sorted(log_path.rglob("*.log")):
        print(f"  {f}")

    events = list(log_path.rglob("events.*"))
    if events:
        print(f"\nTensorBoard event files found: {len(events)}")
        for e in events[:3]:
            print(f"  {e}")


if __name__ == "__main__":
    parser = parse_args()

    print("=" * 60)
    print("Visualization Tool for Multi-Agent RL")
    print("=" * 60)

    # Check available logs
    check_logs(parser.log_dir)

    # Plot training curves
    print("\n[1] Generate training curves from TensorBoard logs")
    if input("Generate plots? (y/n): ").lower() == 'y':
        plot_training_curves(parser.log_dir)

    # Render trained model
    print("\n[2] Render trained model (requires model checkpoint)")
    if input("Render episodes? (y/n): ").lower() == 'y':
        render_trained_model(parser.model_dir, episodes=parser.episodes)

    print("\nDone!")