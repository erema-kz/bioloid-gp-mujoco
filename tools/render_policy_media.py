from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from stable_baselines3 import PPO

from bioloid_sim.exact_env import ExactBioloidGPEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a trained exact-model policy to GitHub-ready PNG and GIF media"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument("--goal-distance", type=float, default=0.5)
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = ExactBioloidGPEnv(
        render_mode="rgb_array",
        episode_seconds=12.0,
        task="walk",
        goal_distance=args.goal_distance,
    )
    policy = PPO.load(args.checkpoint)
    observation, _ = env.reset(seed=args.seed)

    frames: list[Image.Image] = []
    distances: list[float] = []
    capture_every = max(1, round(env.metadata["render_fps"] / args.gif_fps))
    info = {"distance": 0.0, "success": False}
    step = 0
    while step < env.max_steps:
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        if step % capture_every == 0 or terminated or truncated:
            rgb = env.render()
            frame = Image.fromarray(rgb).resize((480, 360), Image.Resampling.LANCZOS)
            frames.append(frame)
            distances.append(float(info["distance"]))
        step += 1
        if terminated or truncated:
            break
    env.close()

    if not frames:
        raise RuntimeError("No frames were rendered")
    if not info.get("success", False):
        raise RuntimeError(
            f"Policy did not reach {args.goal_distance:.2f} m; ended at {info['distance']:.3f} m"
        )

    still_indices = {
        "start": 0,
        "mid_step": min(range(len(distances)), key=lambda i: abs(distances[i] - 0.25)),
        "goal": len(frames) - 1,
    }
    for name, index in still_indices.items():
        frames[index].save(args.output_dir / f"bioloid_{name}.png", optimize=True)

    duration_ms = round(1000 / args.gif_fps)
    frames[0].save(
        args.output_dir / "bioloid_walk.gif",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(
        f"Rendered {len(frames)} frames, reached {info['distance']:.3f} m in "
        f"{step * env.control_dt:.2f} s"
    )


if __name__ == "__main__":
    main()
