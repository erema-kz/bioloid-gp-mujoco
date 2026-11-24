from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from bioloid_sim.exact_env import ExactBioloidGPEnv


_ENV: ExactBioloidGPEnv | None = None
_DIRECTION = "front"
_STEPS_PER_KEYFRAME = 35


def _init_worker(direction: str, steps_per_keyframe: int) -> None:
    global _ENV, _DIRECTION, _STEPS_PER_KEYFRAME
    _DIRECTION = direction
    _STEPS_PER_KEYFRAME = steps_per_keyframe
    _ENV = ExactBioloidGPEnv(
        task="walk",
        goal_distance=100.0,
        episode_seconds=12.0,
        allow_recovery=True,
    )


def _score(candidate: np.ndarray) -> float:
    assert _ENV is not None
    keyframes = candidate.reshape(-1, 16)
    _ENV.reset(seed=123, options={"start_fallen": True, "fall_direction": _DIRECTION})
    total_steps = len(keyframes) * _STEPS_PER_KEYFRAME
    best_score = -1e9
    final_score = -1e9
    for step in range(total_steps):
        phase = step / _STEPS_PER_KEYFRAME
        low = min(int(phase), len(keyframes) - 1)
        high = min(low + 1, len(keyframes) - 1)
        blend = phase - int(phase)
        action = (1.0 - blend) * keyframes[low] + blend * keyframes[high]
        _, _, terminated, truncated, info = _ENV.step(action)
        height_score = float(np.clip((info["height"] - 0.10) / 0.18, 0.0, 1.0))
        upright_score = float(np.clip(info["upright"] / 0.80, 0.0, 1.0))
        pose_score = 12.0 * height_score + 5.0 * upright_score
        best_score = max(best_score, pose_score)
        final_score = pose_score
        if info["recovered"]:
            return 200.0 + 0.2 * (total_steps - step)
        if terminated or truncated:
            break
    return best_score + 0.5 * final_score


def optimize_direction(
    direction: str,
    population: int,
    iterations: int,
    keyframes: int,
    steps_per_keyframe: int,
    workers: int,
    seed: int,
    initial_candidates: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    dimension = keyframes * 16
    mean = np.zeros(dimension, dtype=np.float64)
    std = np.full(dimension, 0.75, dtype=np.float64)
    elite_count = max(4, population // 8)
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(direction, steps_per_keyframe),
    ) as pool:
        best_candidate = mean.copy()
        best_score = -np.inf
        if initial_candidates is not None:
            mirror_scores = np.asarray(
                pool.map(_score, list(initial_candidates)), dtype=np.float64
            )
            mirror_best = int(np.argmax(mirror_scores))
            mean = initial_candidates[mirror_best].copy()
            std.fill(0.30)
            best_candidate = mean.copy()
            best_score = float(mirror_scores[mirror_best])
            print(
                f"{direction} mirrored_seed best={best_score:.2f} "
                f"mean={np.mean(mirror_scores):.2f}",
                flush=True,
            )
        for iteration in range(iterations):
            candidates = np.clip(
                rng.normal(mean, std, size=(population, dimension)), -1.0, 1.0
            )
            scores = np.asarray(pool.map(_score, list(candidates)), dtype=np.float64)
            elite_indices = np.argsort(scores)[-elite_count:]
            elites = candidates[elite_indices]
            iteration_best = int(np.argmax(scores))
            if scores[iteration_best] > best_score:
                best_score = float(scores[iteration_best])
                best_candidate = candidates[iteration_best].copy()
            mean = 0.75 * mean + 0.25 * np.mean(elites, axis=0)
            std = np.maximum(0.08, 0.75 * std + 0.25 * np.std(elites, axis=0))
            print(
                f"{direction} iteration={iteration + 1}/{iterations} "
                f"best={best_score:.2f} mean={np.mean(scores):.2f}",
                flush=True,
            )
            if best_score >= 200.0:
                break
    return best_candidate.reshape(keyframes, 16), best_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize get-up actuator trajectories with CEM")
    parser.add_argument("--output", type=Path, default=Path("runs/getup_trajectories.json"))
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--keyframes", type=int, default=6)
    parser.add_argument("--steps-per-keyframe", type=int, default=35)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--mirror-from",
        choices=("front", "back", "left", "right"),
        help="seed optimization by mirroring an existing direction",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=("front", "back", "left", "right"),
        default=("front", "back", "left", "right"),
    )
    args = parser.parse_args()

    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if int(result["steps_per_keyframe"]) != args.steps_per_keyframe:
            raise ValueError("Existing output uses a different steps-per-keyframe value")
    else:
        result = {
            "steps_per_keyframe": args.steps_per_keyframe,
            "control_dt": 0.02,
            "directions": {},
        }
    for offset, direction in enumerate(args.directions):
        if direction in result["directions"] and not args.overwrite:
            print(f"Skipping existing direction: {direction}")
            continue
        initial_candidates = None
        if args.mirror_from:
            source = np.asarray(
                result["directions"][args.mirror_from]["keyframes"], dtype=np.float64
            )
            if source.shape != (args.keyframes, 16):
                raise ValueError("Mirrored source has a different keyframe shape")
            pairs = ((0, 3), (1, 4), (2, 5), (6, 11), (7, 12), (8, 13), (9, 14), (10, 15))
            mirrored_variants = []
            for mask in range(1 << len(pairs)):
                mirrored = source.copy()
                for pair_index, (right, left) in enumerate(pairs):
                    sign = -1.0 if mask & (1 << pair_index) else 1.0
                    mirrored[:, right] = sign * source[:, left]
                    mirrored[:, left] = sign * source[:, right]
                mirrored_variants.append(mirrored.reshape(-1))
            initial_candidates = np.asarray(mirrored_variants)
        trajectory, score = optimize_direction(
            direction,
            args.population,
            args.iterations,
            args.keyframes,
            args.steps_per_keyframe,
            args.workers,
            args.seed + offset,
            initial_candidates,
        )
        result["directions"][direction] = {
            "score": score,
            "keyframes": trajectory.tolist(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved trajectories to {args.output}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
