"""
XuanCe Compatible Grid Exploration Environment
===============================================
Register the GridExplorationEnv with xuance framework.
"""

import numpy as np
from gymnasium.spaces import Box, Discrete
from xuance.environment import RawMultiAgentEnv


class GridExplorationMAEnv(RawMultiAgentEnv):
    """
    Multi-agent wrapper for GridExplorationEnv.

    This makes the grid exploration environment compatible with xuance's
    MARL training infrastructure.
    """

    def __init__(self, config):
        super().__init__()

        # Import the grid environment
        from environments.grid_exploration import GridExplorationEnv

        # Configuration - handle both Namespace and dict
        self.env_id = "grid_exploration"
        if hasattr(config, 'num_agents'):
            self.num_drones = config.num_agents
            self.grid_size = config.grid_size
            self.max_episode_steps = config.max_episode_steps
            env_seed = config.env_seed if hasattr(config, 'env_seed') else 1
        else:
            self.num_drones = config.get("num_agents", 3)
            self.grid_size = config.get("grid_size", 15)
            self.max_episode_steps = config.get("max_episode_steps", 300)
            env_seed = config.get("env_seed", 1)

        # Create the underlying environment
        self.env = GridExplorationEnv(
            num_agents=self.num_drones,
            grid_size=self.grid_size,
            max_steps=self.max_episode_steps,
            seed=env_seed,
        )

        # Agent info
        self.agents = self.env.agents.copy()
        self.num_agents = len(self.agents)
        self.agents_ids = [f"agent_{i}" for i in range(self.num_agents)]
        # Required by xuance
        self.agent_groups = [self.agents_ids]

        # Spaces
        obs_shape = (4,)  # Each agent's observation: [normalized_x, normalized_y, coverage_percent, nearby_agents]
        act_shape = ()  # Discrete action

        self.observation_space = {
            agent: Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
            for agent in self.agents
        }
        self.action_space = {
            agent: Discrete(5)  # up, down, left, right, stay
            for agent in self.agents
        }
        # Global state space (for centralized training)
        self.state_space = Box(-np.inf, np.inf, shape=(4 * self.num_agents,), dtype=np.float32)

        # Episode tracking
        self._episode_step = 0
        self.max_cycles = self.max_episode_steps
        self.max_episode_steps = self.max_episode_steps  # Required by xuance

    def reset(self):
        """Reset the environment."""
        observations, infos = self.env.reset()
        self._episode_step = 0

        # Build episode_score dict (xuance expects this at top level)
        episode_scores = {agent: float(infos.get(agent, {}).get("episode_score", 0.0)) for agent in self.agents}

        # Convert to xuance format - episode_score must be at top level
        obs_dict = {k: v for k, v in observations.items()}
        info_dict = {
            "infos": infos,
            "episode_step": self._episode_step,
            "episode_score": episode_scores,  # xuance expects this at top level
        }

        return obs_dict, info_dict

    def step(self, actions):
        """Execute one step."""
        # Convert to dict format expected by env
        actions_dict = {
            f"agent_{i}": actions.get(i, 4)
            for i in range(self.num_agents)
        }

        obs, rewards, terminated, truncated, info = self.env.step(actions_dict)

        # Save last observation for state()
        self._last_obs = obs

        self._episode_step += 1

        # Build episode_score dict (xuance expects this at top level)
        episode_scores = {agent: float(info.get(agent, {}).get("episode_score", 0.0)) for agent in self.agents}

        # Convert to xuance format - episode_score must be at top level
        obs_dict = {k: v for k, v in obs.items()}
        rewards_dict = {k: float(v) for k, v in rewards.items()}
        terminated_dict = {k: bool(v) for k, v in terminated.items()}
        # truncated should be a BOOLEAN (not dict) for xuance DummyVecMultiAgentEnv compatibility
        # True if any agent is terminated OR episode step reached max_cycles
        truncated = any(terminated_dict.values()) or self._episode_step >= self.max_cycles

        info_dict = {
            "infos": info,
            "episode_step": self._episode_step,
            "episode_score": episode_scores,  # xuance expects this at top level
        }

        return obs_dict, rewards_dict, terminated_dict, truncated, info_dict

    def render(self, *args, **kwargs):
        """Render the environment."""
        self.env.render()
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def close(self):
        """Close the environment."""
        self.env.close()

    def state(self):
        """Get global state by concatenating all agent observations."""
        # Get latest observations
        obs, _ = self.reset() if not hasattr(self, '_last_obs') else (self._last_obs, {})
        # Concatenate all agent observations
        return np.concatenate([obs[agent] for agent in self.agents])

    def agent_mask(self):
        """Return agent mask (all agents active)."""
        return {agent: True for agent in self.agents}

    def avail_actions(self):
        """Return available actions for each agent."""
        return {
            agent: np.ones(5, dtype=np.bool_)  # All 5 actions available
            for agent in self.agents
        }


# Registry for xuance
REGISTRY = {
    "grid_exploration": GridExplorationMAEnv,
}


if __name__ == "__main__":
    # Test the environment
    from argparse import Namespace

    config = Namespace(
        num_agents=3,
        grid_size=15,
        max_episode_steps=200,
        env_seed=42,
    )

    env = GridExplorationMAEnv(config)

    print("Testing GridExplorationMAEnv...")
    print(f"Number of agents: {env.num_agents}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"\nInitial obs for agent_0: {obs['agent_0']}")

    print("\nRunning 5 steps...")
    for i in range(5):
        actions = {agent: 4 for agent in env.agents}  # stay
        obs, rewards, terms, truncs, info = env.step(actions)
        print(f"Step {i}: rewards={rewards}")

    env.render()
    env.close()
    print("\nTest complete!")