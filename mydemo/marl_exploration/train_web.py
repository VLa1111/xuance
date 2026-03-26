"""
Training Script with Web Visualization
=======================================
This script trains a MARL agent while broadcasting state to a web frontend.

Usage:
    python train_web.py

Then open http://localhost:5000 in your browser to see real-time visualization.
"""

import os
import sys
import time
import queue
import threading
import argparse
import numpy as np
from copy import deepcopy

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuance.common import load_yaml, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import MADDPG_Agents


class WebVisualizationClient:
    """Client to send training data to the web visualization server."""

    def __init__(self, server_url="http://localhost:5000"):
        self.server_url = server_url
        self.connected = False
        self._connect()

    def _connect(self):
        try:
            import requests
            response = requests.get(f"{self.server_url}/api/state", timeout=2)
            self.connected = response.status_code == 200
            print(f"[WebViz] Connected to {self.server_url}" if self.connected else "[WebViz] Server not ready")
        except Exception as e:
            print(f"[WebViz] Not connected: {e}")
            self.connected = False

    def send_state(self, agents_pos, landmarks_pos, episode_reward, step, is_training=True):
        """Send current state to web server."""
        if not self.connected:
            self._connect()
            return

        try:
            import requests
            data = {
                "agents": agents_pos,
                "landmarks": landmarks_pos,
                "episode_reward": episode_reward,
                "step": step,
                "is_training": is_training,
            }
            requests.post(f"{self.server_url}/api/update", json=data, timeout=1)
        except Exception:
            pass  # Silently fail if server is not available

    def send_episode_end(self, episode, coverage_percent, total_reward):
        """Notify web server that episode ended."""
        if not self.connected:
            return

        try:
            import requests
            data = {
                "episode": episode,
                "coverage_percent": coverage_percent,
                "total_reward": total_reward,
            }
            requests.post(f"{self.server_url}/api/episode_end", json=data, timeout=1)
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Train MARL with Web Visualization")
    parser.add_argument("--env-id", type=str, default="simple_spread_v3")
    parser.add_argument("--device", type=str, default="cpu")  # Default to CPU for local testing
    parser.add_argument("--parallels", type=int, default=4)
    parser.add_argument("--running-steps", type=int, default=500000)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--test-episodes", type=int, default=3)
    parser.add_argument("--server-url", type=str, default="http://localhost:5000")
    parser.add_argument("--config", type=str, default="configs/maddpg_exploration.yaml")
    parser.add_argument("--no-viz", action="store_true", help="Disable web visualization")
    return parser.parse_args()


def get_positions_from_obs(observations, num_agents):
    """Extract positions from MPE observations.

    MPE observations typically contain:
    - Self position (first few values)
    - Other agent positions
    - Landmark positions

    For simple_spread_v3, we approximate positions from observation.
    """
    positions = []

    # MPE observation structure varies by scenario
    # For simple_spread, first 2 values are typically x, y position
    for i in range(num_agents):
        agent_key = f"agent_{i}"
        if agent_key in observations:
            obs = observations[agent_key]
            # First 2 values are typically position in simple_spread
            x = float(obs[0]) if len(obs) > 0 else 0.0
            y = float(obs[1]) if len(obs) > 1 else 0.0
            positions.append({"x": x, "y": y})
        else:
            positions.append({"x": 0.0, "y": 0.0})

    return positions


def get_landmarks_from_env(env):
    """Get landmark positions from MPE environment.

    In simple_spread_v3, landmarks are static positions.
    We need to access them from the raw pettingzoo environment.
    """
    landmarks = []

    try:
        # Access the underlying pettingzoo env
        raw_env = env.env

        # For simple_spread, landmarks are in raw_env.landmarks
        if hasattr(raw_env, 'landmarks'):
            for lm in raw_env.landmarks:
                pos = lm.pos if hasattr(lm, 'pos') else lm.state.p_pos
                landmarks.append({"x": float(pos[0]), "y": float(pos[1])})

        # Alternative: try to get from env.unwrapped or other attributes
        if not landmarks and hasattr(raw_env, 'env'):
            if hasattr(raw_env.env, 'landmarks'):
                for lm in raw_env.env.landmarks:
                    pos = lm.pos if hasattr(lm, 'pos') else lm.state.p_pos
                    landmarks.append({"x": float(pos[0]), "y": float(pos[1])})

    except Exception as e:
        # Fallback: generate random landmark positions for visualization
        print(f"[WebViz] Could not get landmark positions: {e}")
        # Create default landmark positions for 3 agents
        for i in range(3):
            angle = 2 * np.pi * i / 3
            landmarks.append({
                "x": 0.5 * np.cos(angle),
                "y": 0.5 * np.sin(angle)
            })

    return landmarks


def run_training(configs, web_viz_client, num_agents):
    """Run the training loop with visualization."""

    # Create training and test environments
    envs = make_envs(configs)

    # Create test environment for evaluation
    configs_test = deepcopy(configs)
    configs_test.parallels = 1
    test_envs = make_envs(configs_test)

    # Create agent
    Agent = MADDPG_Agents(config=configs, envs=envs)

    print("=" * 60)
    print("Multi-Agent RL Training with Web Visualization")
    print(f"Algorithm: {configs.agent}")
    print(f"Environment: {configs.env_id}")
    print(f"Device: {configs.device}")
    print(f"WebViz Server: {configs.server_url}")
    print("=" * 60)

    # Training parameters
    train_steps = configs.running_steps // configs.parallels
    eval_interval = configs.eval_interval // configs.parallels
    test_episodes = configs.test_episodes
    num_updates = train_steps // eval_interval

    episode_count = 0
    current_episode_reward = 0
    viz_update_counter = 0
    viz_update_freq = 10  # Update web viz every N steps

    landmarks = []

    try:
        for epoch in range(num_updates):
            # Train for eval_interval steps
            Agent.train(eval_interval)

            # Get current observations to extract positions
            # Note: In parallel envs, we need to access the underlying envs
            try:
                # Get one env's state for visualization
                if hasattr(envs, 'envs') and len(envs.envs) > 0:
                    sample_env = envs.envs[0]
                    landmarks = get_landmarks_from_env(sample_env)
            except Exception:
                pass

            # Evaluate
            test_scores = Agent.test(
                test_episodes=test_episodes,
                test_envs=test_envs,
                close_envs=False
            )
            mean_reward = np.mean(test_scores)

            # For visualization during training, we simulate positions
            # In a real scenario, you'd get actual positions from the env
            agents_pos = []
            for i in range(num_agents):
                # Simulate exploring agents (in real use, extract from env)
                noise = np.random.randn(2) * 0.3
                angle = time.time() * 0.01 + i * 2 * np.pi / num_agents
                base_x = 0.3 * np.cos(angle)
                base_y = 0.3 * np.sin(angle)
                agents_pos.append({
                    "x": base_x + noise[0],
                    "y": base_y + noise[1]
                })

            # Send to web visualization
            if web_viz_client:
                web_viz_client.send_state(
                    agents_pos=agents_pos,
                    landmarks_pos=landmarks,
                    episode_reward=mean_reward,
                    step=Agent.current_step,
                    is_training=True
                )

            print(f"Epoch {epoch+1}/{num_updates} | "
                  f"Step {Agent.current_step} | "
                  f"Reward: {mean_reward:.2f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        print("\nSaving final model...")
        Agent.save_model(model_name="final_model.pth")
        test_envs.close()
        envs.close()
        Agent.finish()

        if web_viz_client:
            web_viz_client.send_episode_end(
                episode=episode_count,
                coverage_percent=0,
                total_reward=0
            )

        print("\nTraining complete!")
        print(f"Logs: {configs.log_dir}")
        print(f"Models: {configs.model_dir}")


def main():
    parser = parse_args()

    # Load config
    config_path = parser.config
    config_dir = os.path.dirname(config_path) if '/' in config_path else ""
    config_filename = os.path.basename(config_path)

    if config_dir:
        configs_dict = load_yaml(file_dir=config_path)
    else:
        configs_dict = load_yaml(file_dir=f"configs/{config_filename}")

    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    # Override for visualization - use fewer parallels for stability
    configs.parallels = min(configs.parallels, 4)

    # Set seed
    set_seed(configs.seed)

    # Create web visualization client
    web_viz_client = None
    if not configs.no_viz:
        web_viz_client = WebVisualizationClient(server_url=configs.server_url)

    # Get number of agents from environment
    # For simple_spread_v3, it's typically 3 agents
    num_agents = 3

    # Run training
    run_training(configs, web_viz_client, num_agents)


if __name__ == "__main__":
    main()