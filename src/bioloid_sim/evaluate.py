from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from bioloid_sim.env import BioloidGPEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained Bioloid GP policy")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--headless", action="store_true", help="evaluate without opening a window")
    parser.add_argument("--episodes", type=int, default=0, help="stop after N episodes; 0 runs forever")
    args = parser.parse_args()

    env = BioloidGPEnv(render_mode=None if args.headless else "human", episode_seconds=60.0)
    policy = PPO.load(args.checkpoint)
    observation, _ = env.reset()
    episode_return = 0.0
    episode_length = 0
    returns: list[float] = []
    lengths: list[int] = []
    speeds: list[float] = []
    commands: list[float] = []
    speed_sum = 0.0
    command_sum = 0.0
    episode_limit = args.episodes or (10 if args.headless else 0)
    while episode_limit == 0 or len(returns) < episode_limit:
        action, _ = policy.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        episode_length += 1
        speed_sum += info["forward_velocity"]
        command_sum += info["command_speed"]
        if terminated or truncated:
            returns.append(episode_return)
            lengths.append(episode_length)
            speeds.append(speed_sum / episode_length)
            commands.append(command_sum / episode_length)
            observation, _ = env.reset()
            episode_return = 0.0
            episode_length = 0
            speed_sum = 0.0
            command_sum = 0.0
    env.close()
    print(
        f"episodes={len(returns)} mean_return={np.mean(returns):.1f} "
        f"mean_length={np.mean(lengths):.1f} steps "
        f"mean_speed={np.mean(speeds):.3f} m/s mean_command={np.mean(commands):.3f} m/s"
    )


if __name__ == "__main__":
    main()
