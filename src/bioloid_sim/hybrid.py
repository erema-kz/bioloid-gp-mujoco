from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from bioloid_sim.exact_env import ExactBioloidGPEnv


FALL_DIRECTIONS = {
    "left": (0, 1.0),
    "right": (0, -1.0),
    "front": (1, 1.0),
    "back": (1, -1.0),
}


def reference_fall_quaternions(home: np.ndarray) -> dict[str, np.ndarray]:
    import mujoco

    references: dict[str, np.ndarray] = {}
    for name, (axis_index, sign) in FALL_DIRECTIONS.items():
        axis = np.zeros(3, dtype=np.float64)
        axis[axis_index] = 1.0
        rotation = np.empty(4, dtype=np.float64)
        orientation = np.empty(4, dtype=np.float64)
        mujoco.mju_axisAngle2Quat(rotation, axis, sign * np.pi / 2)
        mujoco.mju_mulQuat(orientation, rotation, home)
        references[name] = orientation
    return references


def nearest_fall_direction(
    orientation: np.ndarray,
    references: dict[str, np.ndarray],
    available: set[str],
) -> str:
    return max(
        available,
        key=lambda name: abs(float(np.dot(orientation, references[name]))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk continuously and get up after a fall")
    parser.add_argument("walking_checkpoint", type=Path)
    parser.add_argument("getup_trajectories", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--episode-seconds", type=float, default=120.0)
    parser.add_argument("--goal-distance", type=float, default=100.0)
    parser.add_argument("--start-fallen", action="store_true")
    parser.add_argument(
        "--fall-direction", choices=tuple(FALL_DIRECTIONS), default="front"
    )
    parser.add_argument(
        "--getup-direction",
        choices=tuple(FALL_DIRECTIONS),
        help="force one stored trajectory instead of selecting by orientation",
    )
    args = parser.parse_args()

    trajectory_data = json.loads(args.getup_trajectories.read_text(encoding="utf-8"))
    steps_per_keyframe = int(trajectory_data["steps_per_keyframe"])
    trajectories = {
        name: np.asarray(value["keyframes"], dtype=np.float64)
        for name, value in trajectory_data["directions"].items()
    }
    if not trajectories:
        raise ValueError("The trajectory file contains no directions")

    env = ExactBioloidGPEnv(
        render_mode=None if args.headless else "human",
        episode_seconds=args.episode_seconds,
        task="walk",
        goal_distance=args.goal_distance,
        allow_recovery=True,
    )
    walking_policy = PPO.load(args.walking_checkpoint)
    references = reference_fall_quaternions(env.model.key_qpos[0, 3:7])
    reset_options = (
        {"start_fallen": True, "fall_direction": args.fall_direction}
        if args.start_fallen
        else None
    )

    observation, reset_info = env.reset(options=reset_options)
    mode = "getup" if reset_info.get("start_fallen", False) else "walk"
    direction = args.getup_direction or args.fall_direction
    trajectory_step = 0
    fallen_steps = 0
    ready_steps = 0
    episode_getup = False
    episode_distance = 0.0
    episode_count = 0
    getup_count = 0
    distances: list[float] = []
    limit = args.episodes or (10 if args.headless else 0)

    while limit == 0 or episode_count < limit:
        height = float(env.data.xpos[env.base_id, 2])
        upright = float(env.data.xmat[env.base_id, 7])
        ready_to_walk = height > 0.20 and upright > 0.48
        clearly_fallen = height < 0.19 or upright < 0.35

        if mode == "getup":
            ready_steps = ready_steps + 1 if ready_to_walk else 0
            if ready_steps >= 5:
                mode = "walk"
                episode_getup = True
                fallen_steps = 0
        elif mode == "walk":
            fallen_steps = fallen_steps + 1 if clearly_fallen else 0
            if fallen_steps >= 8:
                mode = "getup"
                direction = args.getup_direction or nearest_fall_direction(
                    env.data.qpos[3:7], references, set(trajectories)
                )
                trajectory_step = 0
                ready_steps = 0

        if mode == "walk":
            action, _ = walking_policy.predict(observation, deterministic=True)
        else:
            if direction not in trajectories:
                direction = next(iter(trajectories))
            keyframes = trajectories[direction]
            phase = trajectory_step / steps_per_keyframe
            low = min(int(phase), len(keyframes) - 1)
            high = min(low + 1, len(keyframes) - 1)
            blend = phase - int(phase)
            action = (1.0 - blend) * keyframes[low] + blend * keyframes[high]
            trajectory_step = (trajectory_step + 1) % (len(keyframes) * steps_per_keyframe)

        observation, _, terminated, truncated, info = env.step(action)
        episode_distance = float(info["distance"])
        if terminated or truncated:
            episode_count += 1
            getup_count += int(episode_getup)
            distances.append(episode_distance)
            observation, reset_info = env.reset(options=reset_options)
            mode = "getup" if reset_info.get("start_fallen", False) else "walk"
            direction = args.getup_direction or args.fall_direction
            trajectory_step = 0
            fallen_steps = 0
            ready_steps = 0
            episode_getup = False

    env.close()
    print(
        f"episodes={episode_count} getup_rate={getup_count / episode_count:.0%} "
        f"mean_distance={np.mean(distances):.3f} m"
    )


if __name__ == "__main__":
    main()
