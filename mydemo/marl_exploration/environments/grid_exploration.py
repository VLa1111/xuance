"""
Grid Exploration Environment for Multi-Agent RL
==============================================
A 2D grid world where agents must explore and cover the map while avoiding obstacles.

Reward Design:
- +10 for covering a new cell
- -0.1 for revisiting an already covered cell
- -5 for hitting an obstacle
- +100 bonus when 100% coverage is achieved

State/Observation:
- Agent's current position (normalized)
- Agent's normalized direction to nearest obstacle
- Coverage grid (which cells have been visited)
- Other agents' positions (for collaboration)

Action:
- Discrete: 5 actions (up, down, left, right, stay)
"""

import numpy as np
from gymnasium.spaces import Box, Discrete
from typing import Optional, Dict, Tuple


class GridExplorationEnv:
    """2D Grid World Exploration Environment with Obstacles for Multiple Agents."""

    def __init__(
        self,
        num_agents: int = 3,
        grid_size: int = 15,
        max_steps: int = 300,
        view_range: int = 3,
        obstacle_density: float = 0.15,  # 15% of cells are obstacles
        seed: Optional[int] = None,
    ):
        """
        Initialize the grid exploration environment.

        Args:
            num_agents: Number of agents (explorers)
            grid_size: Size of the square grid (grid_size x grid_size)
            max_steps: Maximum steps per episode
            view_range: How far each agent can see
            obstacle_density: Fraction of cells that are obstacles (0.0 to 0.3)
            seed: Random seed
        """
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.view_range = view_range
        self.obstacle_density = obstacle_density
        self.np_random = np.random.RandomState(seed)

        # Define spaces - each agent has 5-dim observation
        # [normalized_x, normalized_y, obstacle_direction, coverage_percent, nearby_agents]
        self.observation_space = Box(
            low=-1, high=1,
            shape=(5,),  # [x, y, obstacle_dir, coverage, nearby_agents]
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

        # Obstacle grid (0 = free, 1 = obstacle)
        self.obstacle_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Track visited cells per agent
        self.visited_cells = [set() for _ in range(self.num_agents)]

        # Episode step counter
        self.step_count = 0

        # Episode total reward
        self.episode_rewards = {agent: 0.0 for agent in self.agents}

    def _generate_obstacles(self, num_obstacles: int, min_dist_from_agents: int = 2):
        """Generate random obstacles, avoiding agent starting positions."""
        obstacles_placed = 0
        attempts = 0
        max_attempts = num_obstacles * 100

        while obstacles_placed < num_obstacles and attempts < max_attempts:
            x = self.np_random.randint(0, self.grid_size)
            y = self.np_random.randint(0, self.grid_size)

            # Check if too close to any agent
            too_close = False
            for pos in self.agent_positions:
                if np.linalg.norm(pos - np.array([x, y])) < min_dist_from_agents:
                    too_close = True
                    break

            # Check if cell is already an obstacle
            if self.obstacle_grid[y, x] == 1 or too_close:
                attempts += 1
                continue

            self.obstacle_grid[y, x] = 1
            obstacles_placed += 1
            attempts += 1

    def _get_obstacle_direction(self, pos: np.ndarray) -> float:
        """Get normalized direction to nearest obstacle (0-1 scale).
        Returns 0 if no obstacle within view_range, 1 if obstacle is very close in preferred direction.
        Actually returns the minimum normalized distance to any obstacle (0=obstacle, 1=no obstacle nearby).
        """
        px, py = int(pos[0]), int(pos[1])

        # Check view_range x view_range area around agent
        min_dist = float('inf')
        for dx in range(-self.view_range, self.view_range + 1):
            for dy in range(-self.view_range, self.view_range + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = px + dx, py + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.obstacle_grid[ny, nx] == 1:
                        dist = np.sqrt(dx * dx + dy * dy)
                        min_dist = min(min_dist, dist)

        if min_dist == float('inf'):
            return 1.0  # No obstacle nearby

        # Normalize: max possible distance is view_range * sqrt(2)
        max_dist = self.view_range * np.sqrt(2)
        return min_dist / max_dist

    def reset(self) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        self._reset_internal_state()

        # Random initial positions (avoiding obstacles)
        for i in range(self.num_agents):
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

                if not overlap and self.obstacle_grid[y, x] == 0:
                    self.agent_positions[i] = pos
                    break

        # Generate obstacles after placing agents
        num_obstacles = int(self.grid_size * self.grid_size * self.obstacle_density)
        self._generate_obstacles(num_obstacles)

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

            # Get obstacle direction/distance
            obstacle_dir = self._get_obstacle_direction(pos)

            # Normalized position and coverage info
            obs = np.array([
                pos[0] / self.grid_size,  # normalized x
                pos[1] / self.grid_size,  # normalized y
                obstacle_dir,  # obstacle proximity (1 = safe, 0 = obstacle nearby)
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
                "episode_score": self.episode_rewards[agent],  # xuance framework requires this
            }
            for agent in self.agents
        }

    def get_coverage_percent(self) -> float:
        """Get current coverage percentage (excluding obstacles)."""
        free_cells = self.grid_size * self.grid_size - np.sum(self.obstacle_grid)
        covered_free_cells = np.sum(self.coverage_grid) - np.sum(self.obstacle_grid * self.coverage_grid)
        return float(covered_free_cells / free_cells * 100)

    def get_valid_coverage(self) -> float:
        """Get coverage percentage of valid (non-obstacle) cells."""
        total_free = self.grid_size * self.grid_size - np.sum(self.obstacle_grid)
        if total_free <= 0:
            return 0.0
        covered = np.sum((self.coverage_grid == 1) & (self.obstacle_grid == 0))
        return float(covered / total_free * 100)

    def step(self, actions: Dict) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one step.

        Args:
            actions: Dict mapping agent_id -> action (0-4)

        Returns:
            observations, rewards, terminations, truncations, infos
        """
        # Initialize step rewards
        step_rewards = {agent: 0.0 for agent in self.agents}

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

            # Calculate new position
            new_x = self.agent_positions[i][0] + dx
            new_y = self.agent_positions[i][1] + dy

            # Check bounds
            if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                # Check obstacle
                gx, gy = int(new_x), int(new_y)
                if self.obstacle_grid[gy, gx] == 1:
                    # Hit obstacle - negative reward, no movement
                    step_rewards[agent] = -5.0
                else:
                    # Valid move
                    self.agent_positions[i] = np.array([new_x, new_y], dtype=np.float32)

                    # Check if this is a new cell
                    cell = (gx, gy)
                    is_new_cell = cell not in self.visited_cells[i]

                    if is_new_cell:
                        step_rewards[agent] = 10.0  # New cell exploration reward
                        self.visited_cells[i].add(cell)
                        self.coverage_grid[gy, gx] = 1
                    else:
                        step_rewards[agent] = -0.1  # Penalty for revisiting
            else:
                # Out of bounds - slight penalty
                step_rewards[agent] = -0.5

            # Update episode rewards
            self.episode_rewards[agent] += step_rewards[agent]

        # Mark all current positions as covered
        self._mark_coverage()

        self.step_count += 1

        # Check termination (coverage of valid cells)
        valid_coverage = self.get_valid_coverage()
        all_covered = valid_coverage >= 100.0
        timeout = self.step_count >= self.max_steps

        # Use immediate step_rewards for learning
        rewards = step_rewards

        # Terminations (done if fully covered valid cells)
        terminations = {agent: all_covered for agent in self.agents}

        # Truncations (timeout)
        truncations = {agent: timeout for agent in self.agents}

        observations = self._get_observations()
        infos = self._get_infos()

        return observations, rewards, terminations, truncations, infos

    def compute_reward(self) -> float:
        """
        Compute team reward based on exploration metrics.
        """
        coverage = self.get_valid_coverage()

        # Base reward from coverage
        reward = coverage * 0.1

        # Speed bonus
        if self.step_count > 0:
            reward += coverage / self.step_count * 10

        # Completion bonus
        if coverage >= 100.0:
            reward += 100.0

        return float(reward)

    def render(self):
        """Render the grid world as ASCII art (for debugging)."""
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)

        # Mark obstacles
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.obstacle_grid[y, x] == 1:
                    grid[y, x] = 'X'

        # Mark coverage
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.coverage_grid[y, x] == 1 and self.obstacle_grid[y, x] == 0:
                    grid[y, x] = 'o'

        # Mark agents
        for i, pos in enumerate(self.agent_positions):
            gx, gy = int(pos[0]), int(pos[1])
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                if self.obstacle_grid[gy, gx] == 0:  # Don't overwrite obstacle
                    grid[gy, gx] = str(i)

        # Print
        print("-" * (self.grid_size * 2 + 1))
        for row in grid:
            print("|" + "".join(row) + "|")
        print("-" * (self.grid_size * 2 + 1))
        print(f"Coverage: {self.get_valid_coverage():.1f}% | Obstacles: {int(np.sum(self.obstacle_grid))} | Step: {self.step_count}")

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
        return obs, infos

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
    print("Testing GridExplorationEnv with Obstacles...")

    env = GridExplorationEnv(
        num_agents=3,
        grid_size=15,
        max_steps=200,
        obstacle_density=0.15
    )

    obs, info = env.reset()
    print("\nInitial observations:")
    for agent, o in obs.items():
        print(f"  {agent}: {o}")

    env.render()

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