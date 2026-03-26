"""
Grid Exploration Environment for Multi-Agent RL
==============================================
A simple 2D grid world where agents must explore and cover the entire map.

Reward Design:
- +10 for covering a new cell
- -0.1 for revisiting an already covered cell
- +100 bonus when 100% coverage is achieved
- Sparse reward based on coverage percentage

State/Observation:
- Agent's current position
- Agent's previous position
- Coverage grid (which cells have been visited)
- Other agents' positions (for collaboration)

Action:
- Discrete: 5 actions (up, down, left, right, stay)
"""

import numpy as np
from gymnasium.spaces import Box, Discrete
from typing import Optional, Dict, Tuple


class GridExplorationEnv:
    """2D Grid World Exploration Environment for Multiple Agents."""

    def __init__(
        self,
        num_agents: int = 3,
        grid_size: int = 10,
        max_steps: int = 200,
        view_range: int = 3,
        seed: Optional[int] = None,
    ):
        """
        Initialize the grid exploration environment.

        Args:
            num_agents: Number of agents (explorers)
            grid_size: Size of the square grid (grid_size x grid_size)
            max_steps: Maximum steps per episode
            view_range: How far each agent can see (not used in simple version)
            seed: Random seed
        """
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.view_range = view_range
        self.np_random = np.random.RandomState(seed)

        # Define spaces
        self.observation_space = Box(
            low=-1, high=1,
            shape=(num_agents * 4,),  # [x, y, coverage, other_agents_count]
            dtype=np.float32
        )
        self.action_space = Discrete(5)  # 0:up, 1:down, 2:left, 3:right, 4:stay

        # Agent names
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.possible_agents = self.agents.copy()

        # State
        self._reset_internal_state()

    def _reset_internal_state(self):
        """Reset internal state variables."""
        # Agent positions (x, y)
        self.agent_positions = np.zeros((self.num_agents, 2), dtype=np.float32)

        # Coverage grid (0 = not covered, 1 = covered)
        self.coverage_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Track visited cells per agent
        self.visited_cells = [set() for _ in range(self.num_agents)]

        # Episode step counter
        self.step_count = 0

        # Episode total reward
        self.episode_rewards = {agent: 0.0 for agent in self.agents}

    def reset(self) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        self._reset_internal_state()

        # Random initial positions
        for i in range(self.num_agents):
            # Place agents in random positions (not overlapping)
            while True:
                x = self.np_random.randint(0, self.grid_size)
                y = self.np_random.randint(0, self.grid_size)
                pos = np.array([x, y], dtype=np.float32)

                # Check no overlap with existing agents
                overlap = False
                for j in range(i):
                    if np.allclose(self.agent_positions[j], pos):
                        overlap = True
                        break

                if not overlap:
                    self.agent_positions[i] = pos
                    break

        # Mark initial positions as covered
        self._mark_coverage()

        observations = self._get_observations()
        infos = self._get_infos()

        return observations, infos

    def _mark_coverage(self):
        """Mark cells covered by all agents."""
        for pos in self.agent_positions:
            gx = int(np.clip(pos[0], 0, self.grid_size - 1))
            gy = int(np.clip(pos[1], 0, self.grid_size - 1))
            self.coverage_grid[gy, gx] = 1

    def _get_observations(self) -> Dict:
        """Get observations for all agents."""
        observations = {}

        for i, agent in enumerate(self.agents):
            pos = self.agent_positions[i]

            # Count other agents nearby
            other_agents_nearby = 0
            for j, other_pos in enumerate(self.agent_positions):
                if i != j:
                    dist = np.linalg.norm(pos - other_pos)
                    if dist < 3:  # within 3 cells
                        other_agents_nearby += 1

            # Normalized position and coverage info
            obs = np.array([
                pos[0] / self.grid_size,  # normalized x
                pos[1] / self.grid_size,  # normalized y
                self.get_coverage_percent() / 100.0,  # normalized coverage
                other_agents_nearby / self.num_agents,  # normalized nearby agents
            ], dtype=np.float32)

            observations[agent] = obs

        return observations

    def _get_infos(self) -> Dict:
        """Get info dict for all agents."""
        return {
            agent: {
                "coverage": self.get_coverage_percent(),
                "step": self.step_count,
            }
            for agent in self.agents
        }

    def get_coverage_percent(self) -> float:
        """Get current coverage percentage."""
        return float(np.sum(self.coverage_grid) / (self.grid_size * self.grid_size) * 100)

    def step(self, actions: Dict) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one step.

        Args:
            actions: Dict mapping agent_id -> action (0-4)

        Returns:
            observations, rewards, terminations, truncations, infos
        """
        # Process each agent's action
        for i, agent in enumerate(self.agents):
            action = actions.get(agent, 4)  # default: stay

            # Move agent
            dx, dy = 0, 0
            if action == 0:  # up
                dy = 1
            elif action == 1:  # down
                dy = -1
            elif action == 2:  # left
                dx = -1
            elif action == 3:  # right
                dx = 1
            # action == 4: stay

            # Update position with bounds
            new_x = np.clip(self.agent_positions[i][0] + dx, 0, self.grid_size - 1)
            new_y = np.clip(self.agent_positions[i][1] + dy, 0, self.grid_size - 1)
            self.agent_positions[i] = np.array([new_x, new_y], dtype=np.float32)

            # Check if this is a new cell
            gx, gy = int(new_x), int(new_y)
            cell = (gx, gy)
            is_new_cell = cell not in self.visited_cells[i]

            # Compute reward
            reward = 0
            if is_new_cell:
                reward = 10.0  # New cell exploration reward
                self.visited_cells[i].add(cell)
                self.coverage_grid[gy, gx] = 1
            else:
                reward = -0.1  # Penalty for revisiting

            self.episode_rewards[agent] += reward

        # Mark all current positions as covered
        self._mark_coverage()

        self.step_count += 1

        # Check termination
        all_covered = self.get_coverage_percent() >= 100.0
        timeout = self.step_count >= self.max_steps

        # Rewards for all agents (team reward)
        rewards = {
            agent: self.compute_reward() for agent in self.agents
        }

        # Terminations (done if fully covered)
        terminations = {agent: all_covered for agent in self.agents}

        # Truncations (timeout)
        truncations = {agent: timeout for agent in self.agents}

        observations = self._get_observations()
        infos = self._get_infos()

        return observations, rewards, terminations, truncations, infos

    def compute_reward(self) -> float:
        """
        Compute team reward based on exploration metrics.

        Reward Components:
        - Coverage progress: encourage covering more cells
        - Speed bonus: reward faster exploration
        - Completion bonus: huge reward for 100% coverage
        """
        coverage = self.get_coverage_percent()

        # Base reward from coverage
        reward = coverage * 0.1

        # Speed bonus: reward for steps taken
        if self.step_count > 0:
            reward += coverage / self.step_count * 10

        # Completion bonus
        if coverage >= 100.0:
            reward += 100.0

        return float(reward)

    def render(self):
        """Render the grid world as ASCII art (for debugging)."""
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)

        # Mark coverage
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.coverage_grid[y, x] == 1:
                    grid[y, x] = 'o'

        # Mark agents
        for i, pos in enumerate(self.agent_positions):
            gx, gy = int(pos[0]), int(pos[1])
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                grid[gy, gx] = str(i)

        # Print
        print("-" * (self.grid_size * 2 + 1))
        for row in grid:
            print("|" + "".join(row) + "|")
        print("-" * (self.grid_size * 2 + 1))
        print(f"Coverage: {self.get_coverage_percent():.1f}% | Step: {self.step_count}")

    def close(self):
        """Clean up resources."""
        pass


# For PettingZoo compatibility
class PettingZooWrapper:
    """
    Wrapper to make GridExplorationEnv compatible with PettingZoo API.
    This allows it to work with xuance's MARL infrastructure.
    """

    def __init__(self, env: GridExplorationEnv):
        self.env = env
        self.agents = env.agents.copy()
        self.possible_agents = env.possible_agents.copy()
        self._agent_ids = list(self.agents)

    def reset(self):
        obs, infos = self.env.reset()
        return obs

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return obs, rewards, terminations, truncations, infos

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        return {agent: self.env.observation_space for agent in self.agents}

    @property
    def action_space(self):
        return {agent: self.env.action_space for agent in self.agents}


# Test
if __name__ == "__main__":
    print("Testing GridExplorationEnv...")

    env = GridExplorationEnv(num_agents=3, grid_size=10, max_steps=200)

    obs, info = env.reset()
    print("\nInitial observations:")
    for agent, o in obs.items():
        print(f"  {agent}: {o}")

    print("\nTaking 10 random steps...")
    for step in range(10):
        actions = {
            agent: env.np_random.randint(0, 5)
            for agent in env.agents
        }
        obs, rewards, terms, truncs, infos = env.step(actions)
        env.render()
        print(f"Rewards: {rewards}")

    env.close()
    print("\nTest complete!")