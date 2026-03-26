"""
Multi-Agent RL Training with Real-time Visualization
=====================================================
This script demonstrates how to train a MARL agent while visualizing
the training progress in real-time.

Features:
- Real-time reward curve
- Episode length tracking
- Loss monitoring
- Model checkpointing
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from copy import deepcopy

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuance.common import load_yaml, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import MADDPG_Agents


class TrainingVisualizer:
    """Real-time training visualization."""

    def __init__(self, max_points=200):
        self.max_points = max_points
        self.rewards = deque(maxlen=max_points)
        self.steps = deque(maxlen=max_points)
        self.actor_losses = deque(maxlen=max_points)
        self.critic_losses = deque(maxlen=max_points)

        # Set up plotting
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("Multi-Agent RL Training Progress", fontsize=14)

        # Configure subplots
        self.reward_ax = self.axes[0, 0]
        self.reward_ax.set_title("Episode Rewards")
        self.reward_ax.set_xlabel("Training Steps")
        self.reward_ax.set_ylabel("Reward")
        self.reward_ax.grid(True, alpha=0.3)

        self.loss_ax = self.axes[0, 1]
        self.loss_ax.set_title("Losses")
        self.loss_ax.set_xlabel("Training Steps")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.grid(True, alpha=0.3)

        self.steps_ax = self.axes[1, 0]
        self.steps_ax.set_title("Episode Length")
        self.steps_ax.set_xlabel("Training Steps")
        self.steps_ax.set_ylabel("Length")
        self.steps_ax.grid(True, alpha=0.3)

        self.best_ax = self.axes[1, 1]
        self.best_ax.set_title("Best Score")
        self.best_ax.axis('off')

        self.reward_line, = self.reward_ax.plot([], [], 'b-', label='Reward')
        self.reward_mean, = self.reward_ax.plot([], [], 'r--', label='Mean (10)')
        self.actor_line, = self.loss_ax.plot([], [], 'g-', label='Actor Loss')
        self.critic_line, = self.loss_ax.plot([], [], 'm-', label='Critic Loss')
        self.steps_line, = self.steps_ax.plot([], [], 'c-')

        self.best_text = self.best_ax.text(0.5, 0.5, '', fontsize=12,
                                            ha='center', va='center',
                                            transform=self.best_ax.transAxes)

        plt.tight_layout()

        self.best_score = -np.inf

    def update(self, step, reward, actor_loss, critic_loss, episode_len):
        """Update plots with new data."""
        self.steps.append(step)
        self.rewards.append(reward)
        self.actor_losses.append(actor_loss if actor_loss else 0)
        self.critic_losses.append(critic_loss if critic_loss else 0)

        # Update reward plot
        self.reward_line.set_data(list(self.steps), list(self.rewards))
        self.reward_ax.relim()
        self.reward_ax.autoscale_view()

        # Moving average
        if len(self.rewards) > 10:
            mean_reward = np.mean(list(self.rewards)[-10:])
            self.reward_mean.set_data([self.steps[-10], self.steps[-1]], [mean_reward, mean_reward])

        # Update loss plot
        self.actor_line.set_data(list(self.steps), list(self.actor_losses))
        self.critic_line.set_data(list(self.steps), list(self.critic_losses))
        self.loss_ax.relim()
        self.loss_ax.autoscale_view()

        # Update steps plot
        self.steps_line.set_data(list(self.steps), [episode_len] * len(self.steps))
        self.steps_ax.relim()
        self.steps_ax.autoscale_view()

        # Update best score
        if reward > self.best_score:
            self.best_score = reward
        self.best_text.set_text(f"Best Score: {self.best_score:.2f}\nCurrent: {reward:.2f}")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """Close visualization."""
        plt.ioff()
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MARL agent with real-time visualization"
    )
    parser.add_argument("--env-id", type=str, default="simple_spread_v3")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--parallels", type=int, default=8)
    parser.add_argument("--running-steps", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--test-episodes", type=int, default=3)
    parser.add_argument("--config", type=str,
                        default="configs/maddpg_exploration.yaml")
    return parser.parse_args()


def main():
    parser = parse_args()

    # Load config
    config_dir = parser.config.rsplit("/", 1)[0] + "/"
    configs_dict = load_yaml(file_dir=parser.config)
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    # Set seed
    set_seed(configs.seed)

    # Create environments
    envs = make_envs(configs)

    # Create agent
    Agent = MADDPG_Agents(config=configs, envs=envs)

    # Create visualizer
    visualizer = TrainingVisualizer()

    print("=" * 60)
    print("Training Multi-Agent RL for Robot Swarm Exploration")
    print(f"Algorithm: {configs.agent}")
    print(f"Environment: {configs.env_id}")
    print(f"Device: {configs.device}")
    print("=" * 60)

    # Training loop
    train_steps = configs.running_steps // configs.parallels
    eval_interval = configs.eval_interval // configs.parallels
    test_episodes = configs.test_episodes

    # Create test environment
    configs_test = deepcopy(configs)
    configs_test.parallels = 1
    configs_test.render = True
    configs_test.render_mode = 'human'
    test_envs = make_envs(configs_test)

    num_updates = train_steps // eval_interval

    try:
        for epoch in range(num_updates):
            # Train
            Agent.train(eval_interval)

            # Evaluate
            test_scores = Agent.test(
                test_episodes=test_episodes,
                test_envs=test_envs,
                close_envs=False
            )
            mean_reward = np.mean(test_scores)

            # Get losses (approximate from last training batch)
            actor_loss = getattr(Agent, 'last_actor_loss', 0)
            critic_loss = getattr(Agent, 'last_critic_loss', 0)

            # Update visualization
            visualizer.update(
                step=Agent.current_step,
                reward=mean_reward,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                episode_len=eval_interval
            )

            print(f"Epoch {epoch+1}/{num_updates} | "
                  f"Step {Agent.current_step} | "
                  f"Reward: {mean_reward:.2f} | "
                  f"Best: {visualizer.best_score:.2f}")

            # Save best model
            if mean_reward > visualizer.best_score:
                Agent.save_model(model_name="best_model.pth")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        # Final evaluation and save
        print("\nSaving final model...")
        Agent.save_model(model_name="final_model.pth")
        visualizer.close()
        test_envs.close()
        envs.close()
        Agent.finish()

        print("\nTraining complete!")
        print(f"Best score achieved: {visualizer.best_score:.2f}")
        print(f"Logs: {configs.log_dir}")
        print(f"Models: {configs.model_dir}")


if __name__ == "__main__":
    main()