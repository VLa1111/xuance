"""
Quick Start: Multi-Agent RL for Robot Swarm Exploration
========================================================
Minimal example to get started with XuanCe MARL framework.

This runs MADDPG on MPE simple_spread environment.
"""

import xuance


def main():
    print("=" * 60)
    print("Quick Start: Multi-Agent RL with XuanCe")
    print("=" * 60)

    # Method 1: Using get_runner (simplest)
    print("\n[Method 1] Using get_runner() - One-liner启动")
    print("-" * 40)

    runner = xuance.get_runner(
        algo='maddpg',       # Multi-Agent DDPG
        env='mpe',           # Multi-Agent Particle Environment
        env_id='simple_spread_v3',  # N agents cover N landmarks
        device='cuda:0',     # or 'cpu'
    )

    # Train for 1000 steps as a quick demo
    print("\nTraining for 1000 steps (quick demo)...")
    runner.run(mode='train')
    print("Done!")


if __name__ == "__main__":
    main()