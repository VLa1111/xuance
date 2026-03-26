"""
Universal Training Script for Multi-Agent RL
============================================
Train MARL agents with unified interface and optional Web visualization.

Usage:
    # Quick test (no visualization)
    python train_grid.py --config configs/grid_exploration.yaml

    # With Web visualization
    python train_grid.py --config configs/grid_exploration.yaml --visualize

    # Benchmark mode
    python train_grid.py --config configs/grid_exploration.yaml --benchmark

    # MPE environment
    python train_grid.py --config configs/maddpg_exploration.yaml --visualize
"""

import os
import sys
import argparse
import numpy as np
from copy import deepcopy

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Register the custom environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environments import REGISTRY

from xuance.common import load_yaml, recursive_dict_update
from xuance.environment import make_envs
from xuance.environment.multi_agent_env import REGISTRY_MULTI_AGENT_ENV
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import IQL_Agents


class WebVisualizationClient:
    """Client to send training data to the web visualization server."""

    def __init__(self, server_url="http://localhost:5001"):
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

    def send_state(self, agents_pos, landmarks_pos, episode_reward, step, is_training=True, coverage=None):
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
            if coverage is not None:
                data["coverage_percent"] = coverage
            requests.post(f"{self.server_url}/api/update", json=data, timeout=1)
        except Exception:
            pass

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


def get_positions_from_obs(observations, num_agents):
    """Extract agent positions from observations."""
    positions = []
    for i in range(num_agents):
        agent_key = f"agent_{i}"
        if agent_key in observations:
            obs = observations[agent_key]
            # First 2 values are typically position
            x = float(obs[0]) if len(obs) > 0 else 0.0
            y = float(obs[1]) if len(obs) > 1 else 0.0
            positions.append({"x": x, "y": y})
        else:
            positions.append({"x": 0.0, "y": 0.0})
    return positions


def get_grid_positions(env):
    """Extract agent and grid positions from grid exploration environment."""
    positions = []
    coverage = 0.0

    try:
        # Chain: DummyVecEnv -> XuanCeMultiAgentEnvWrapper -> GridExplorationMAEnv -> GridExplorationEnv
        raw_env = env

        # Try to unwrap through layers
        for _ in range(4):  # Max 4 layers of wrapping
            if hasattr(raw_env, 'env'):
                raw_env = raw_env.env
            elif hasattr(raw_env, 'agent_positions'):
                break

        # Get agent positions
        if hasattr(raw_env, 'agent_positions'):
            for pos in raw_env.agent_positions:
                positions.append({"x": float(pos[0]), "y": float(pos[1])})
            print(f"[get_grid_positions] Found {len(positions)} agent positions")

        # Get coverage
        if hasattr(raw_env, 'get_coverage'):
            coverage = raw_env.get_coverage() * 100
            print(f"[get_grid_positions] Coverage: {coverage}")

    except Exception as e:
        print(f"[get_grid_positions] Error: {e}")

    return positions, coverage


def get_mpe_landmarks(env):
    """Get landmark positions from MPE environment."""
    landmarks = []
    try:
        raw_env = env.env
        if hasattr(raw_env, 'landmarks'):
            for lm in raw_env.landmarks:
                pos = lm.pos if hasattr(lm, 'pos') else lm.state.p_pos
                landmarks.append({"x": float(pos[0]), "y": float(pos[1])})
    except Exception:
        pass
    return landmarks


def parse_args():
    parser = argparse.ArgumentParser(description="Train MARL Agents")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--parallels", type=int, default=8)
    parser.add_argument("--running-steps", type=int, default=2000000)
    parser.add_argument("--eval-interval", type=int, default=50000)
    parser.add_argument("--test-episodes", type=int, default=5)
    parser.add_argument("--config", type=str, default="configs/grid_exploration.yaml")
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--visualize", action="store_true", default=False,
                        help="Enable Web visualization")
    parser.add_argument("--server-url", type=str, default="http://localhost:5001",
                        help="Web visualization server URL")
    return parser.parse_args()


def print_config(configs, visualize=False):
    """Print training configuration."""
    print("=" * 60)
    print("Multi-Agent RL Training")
    print("=" * 60)
    print(f"  Algorithm: {configs.agent}")
    print(f"  Environment: {configs.env_name} / {configs.env_id}")
    print(f"  Device: {configs.device}")
    print(f"  Parallels: {configs.parallels}")
    print(f"  Running Steps: {configs.running_steps}")
    if visualize:
        print(f"  Visualization: Enabled ({configs.server_url})")
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
    configs.env_seed = configs.seed

    # Register custom environment with xuance
    REGISTRY_MULTI_AGENT_ENV.update(REGISTRY)

    # Create environments
    envs = make_envs(configs)

    # Determine number of agents
    num_agents = configs.num_agents if hasattr(configs, 'num_agents') else 3

    # Create agent (IQL works well for discrete actions)
    # Use agent from config, default to IQL
    agent_name = configs.agent if hasattr(configs, 'agent') else "IQL"
    try:
        if agent_name == "MADDPG":
            from xuance.torch.agents import MADDPG_Agents
            Agent = MADDPG_Agents
        elif agent_name == "MAPPO":
            from xuance.torch.agents import MAPPO_Agents
            Agent = MAPPO_Agents
        elif agent_name == "QMIX":
            from xuance.torch.agents import QMIX_Agents
            Agent = QMIX_Agents
        else:
            from xuance.torch.agents import IQL_Agents
            Agent = IQL_Agents
    except:
        from xuance.torch.agents import IQL_Agents
        Agent = IQL_Agents

    agent = Agent(config=configs, envs=envs)

    # Initialize visualization client if requested
    web_viz = None
    if parser.visualize:
        web_viz = WebVisualizationClient(server_url=parser.server_url)

    # Print config
    print_config(configs, visualize=parser.visualize)

    # Training loop
    train_steps = configs.running_steps // configs.parallels

    if configs.benchmark:
        print("\n[Benchmark Mode] Train + Evaluate")
        print("-" * 40)

        # Create test environment
        configs_test = deepcopy(configs)
        configs_test.parallels = 1
        configs_test.render = True
        configs_test.render_mode = 'human'
        test_envs = make_envs(configs_test)

        eval_interval = configs.eval_interval // configs.parallels
        test_episode = configs.test_episodes
        num_updates = train_steps // eval_interval

        best_reward = -float('inf')

        for epoch in range(num_updates):
            print(f"\nEpoch {epoch + 1}/{num_updates}")
            print(f"  Training steps: {epoch * eval_interval}/{train_steps}")

            # Train
            agent.train(eval_interval)

            # Evaluate
            test_scores = agent.test(
                test_episodes=test_episode,
                test_envs=test_envs,
                close_envs=False
            )

            mean_reward = np.mean(test_scores) if len(test_scores) > 0 else 0
            print(f"  Evaluation - Mean Reward: {mean_reward:.2f}")

            # Visualize evaluation
            if web_viz and web_viz.connected:
                try:
                    # Get positions from test environment
                    agents_pos = []
                    landmarks_pos = []
                    coverage = None

                    if hasattr(test_envs, 'envs') and len(test_envs.envs) > 0:
                        sample_env = test_envs.envs[0]
                        print(f"[Debug] sample_env type: {type(sample_env)}")

                        # Try to get grid positions
                        agents_pos, coverage = get_grid_positions(sample_env)

                        if agents_pos:
                            landmarks_pos = []
                            print(f"[Debug] Got grid positions: {agents_pos}")
                        else:
                            # MPE environment
                            obs, _ = sample_env.reset()
                            agents_pos = get_positions_from_obs(obs, num_agents)
                            landmarks_pos = get_mpe_landmarks(sample_env)
                            print(f"[Debug] Using MPE positions: {agents_pos}")

                    if agents_pos:
                        web_viz.send_state(
                            agents_pos=agents_pos,
                            landmarks_pos=landmarks_pos,
                            episode_reward=mean_reward,
                            step=agent.current_step,
                            is_training=False,
                            coverage=coverage
                        )
                        print(f"[WebViz] Sent state: {len(agents_pos)} agents, coverage={coverage}")
                    else:
                        print(f"[WebViz] No positions extracted, skipping send")
                except Exception as e:
                    print(f"[WebViz] Error: {e}")

            # Track best
            if mean_reward > best_reward:
                best_reward = mean_reward
                agent.save_model(model_name="best_model.pth")
                print(f"  New best model saved! Reward: {best_reward:.2f}")

        print("\n" + "=" * 60)
        print("Benchmark Complete!")
        print(f"Best Reward: {best_reward:.2f}")
        print(f"Logs: {configs.log_dir}")
        print("=" * 60)

    else:
        print("\n[Training Mode]")
        agent.train(train_steps)
        agent.save_model("final_model.pth")
        print("Training complete!")

    agent.finish()


if __name__ == "__main__":
    main()
