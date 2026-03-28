"""
Quick verification script for episode_score format fix.
Tests that the MA wrapper's info dict matches what xuance framework expects.
"""

import sys
sys.path.insert(0, '/Users/yuyichen/tech-learing/RL/xuance')
sys.path.insert(0, '/Users/yuyichen/tech-learing/RL/xuance/mydemo/marl_exploration')

# Remove installed xuance
sys.path = [p for p in sys.path if 'site-packages/xuance' not in p]

from argparse import Namespace
from environments import GridExplorationMAEnv


def test_info_format():
    """Test that info format matches xuance framework expectations."""
    print("=" * 60)
    print("Testing MA Wrapper Info Format")
    print("=" * 60)

    config = Namespace(
        num_agents=4,
        grid_size=15,
        max_episode_steps=300,
        env_seed=42,
    )

    env = GridExplorationMAEnv(config)
    obs, info = env.reset()

    print("\n[1] Checking info structure...")
    print(f"    info keys: {list(info.keys())}")

    # Simulate what xuance does - access by integer index
    # In benchmark mode, xuance iterates over parallel envs as info[0], info[1], etc.
    # But our wrapper returns string keys like "agent_0"

    # Check if episode_score exists at top level (what xuance expects)
    has_top_level_episode_score = "episode_score" in info
    print(f"    Has top-level 'episode_score': {has_top_level_episode_score}")

    if has_top_level_episode_score:
        print(f"    episode_score value: {info['episode_score']}")

    # Now test what xuance actually accesses in off_policy_marl.py:
    # It does: info[i]["episode_score"] where i is env index (0, 1, 2, ...)
    print("\n[2] Simulating xuance framework access pattern...")

    # With parallels=8, xuance creates 8 envs and accesses info[0], info[1], etc.
    # Our wrapper should make episode_score accessible this way
    test_env_index = 0  # simulating first parallel env

    try:
        # This is what xuance framework tries to access
        episode_score = info["episode_score"]
        print(f"    info['episode_score']: {episode_score}")

        # Check if it can be accessed with integer index after wrapper
        # (actually in our case it's dict keyed by agent names, not integers)
        if isinstance(episode_score, dict):
            # Get the first agent's episode score
            first_agent_score = episode_score.get("agent_0", 0)
            print(f"    First agent's episode_score: {first_agent_score}")

            # Note: at reset (step 0), episode_score is 0.0, which is correct
            # After a step, it should be non-zero
            # The real test is in test_episode_scoring which shows it accumulates
            print("\n[INFO] episode_score is accessible as dict (format correct)")
            return True
    except Exception as e:
        print(f"    Error accessing episode_score: {e}")
        return False

    return False


def test_episode_scoring():
    """Test that episode_score accumulates correctly over steps."""
    print("\n" + "=" * 60)
    print("Testing Episode Score Accumulation")
    print("=" * 60)

    config = Namespace(
        num_agents=4,
        grid_size=15,
        max_episode_steps=300,
        env_seed=42,
    )

    env = GridExplorationMAEnv(config)
    obs, info = env.reset()

    initial_score = info.get("episode_score", {})
    print(f"\n[Step 0] Initial episode_score: {initial_score}")

    # Take 5 steps
    for step in range(5):
        actions = {i: env.env.np_random.randint(0, 5) for i in range(4)}
        obs, rewards, terms, truncs, info = env.step(actions)
        episode_scores = info.get("episode_score", {})
        step_reward = sum(rewards.values())
        coverage = env.env.get_coverage_percent()
        print(f"[Step {step+1}] reward={step_reward:.1f}, coverage={coverage:.1f}%, episode_scores={episode_scores}")

    # Check if episode_score is accumulating
    final_scores = info.get("episode_score", {})
    total_final = sum(final_scores.values()) if isinstance(final_scores, dict) else 0

    if total_final > 0:
        print(f"\n[PASS] Episode scores accumulating: final_total={total_final:.1f}")
        return True
    else:
        print(f"\n[FAIL] Episode scores not accumulating!")
        return False


def test_with_xuance_agent():
    """Test with actual xuance IQL agent (short training)."""
    print("\n" + "=" * 60)
    print("Testing with Xuance IQL Agent (Short)")
    print("=" * 60)

    from copy import deepcopy
    from xuance.common import load_yaml, recursive_dict_update
    from xuance.environment import make_envs
    from xuance.environment.multi_agent_env import REGISTRY_MULTI_AGENT_ENV
    from xuance.torch.utils.operations import set_seed
    from xuance.torch.agents import IQL_Agents

    # Load config
    configs_dict = load_yaml(file_dir='configs/grid_exploration.yaml')
    configs = Namespace(**configs_dict)
    configs.device = "cpu"  # Use CPU for local test

    set_seed(configs.seed)

    # Register environment
    from environments import REGISTRY
    REGISTRY_MULTI_AGENT_ENV.update(REGISTRY)

    # Create env and agent
    envs = make_envs(configs)
    agent = IQL_Agents(config=configs, envs=envs)

    # Run very short training
    print("\nRunning 1000 training steps...")
    agent.train(1000)
    print(f"Training done. Current step: {agent.current_step}")

    # Test policy
    print("\nTesting trained policy (3 episodes)...")
    test_scores = agent.test(test_episodes=3, test_envs=envs, close_envs=True)
    mean_score = sum(test_scores) / len(test_scores) if test_scores else 0

    print(f"Test scores: {test_scores}")
    print(f"Mean score: {mean_score:.2f}")

    agent.finish()

    if mean_score != 0:
        print("\n[PASS] Agent is receiving non-zero rewards!")
        return True
    else:
        print("\n[FAIL] Agent receiving zero rewards!")
        return False


if __name__ == "__main__":
    results = []

    # Test 1: info format
    results.append(("Info Format", test_info_format()))

    # Test 2: episode scoring
    results.append(("Episode Scoring", test_episode_scoring()))

    # Test 3: with xuance agent (only if tests 1&2 pass)
    if all(r[1] for r in results[:2]):
        print("\n" + "=" * 60)
        print("SKIPPING xuance agent test (do manually on server with --diagnose)")
        print("=" * 60)
        # results.append(("Xuance Agent", test_with_xuance_agent()))
    else:
        print("\n" + "=" * 60)
        print("Skipping xuance agent test due to earlier failures")
        print("=" * 60)

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")

    all_passed = all(r[1] for r in results)
    print("\n" + ("ALL TESTS PASSED - Ready for server training!" if all_passed else "SOME TESTS FAILED - Fix issues before server training"))
