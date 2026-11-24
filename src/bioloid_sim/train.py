from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from bioloid_sim.env import BioloidGPEnv
from bioloid_sim.exact_env import ExactBioloidGPEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Bioloid GP with PPO")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--model", choices=("approx", "exact"), default="approx")
    parser.add_argument("--task", choices=("balance", "walk"), default="balance")
    parser.add_argument("--goal-distance", type=float, default=0.5)
    parser.add_argument("--resume", type=Path, help="continue from an existing PPO checkpoint")
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(
            "runs/bioloid_exact_ppo" if args.model == "exact" else "runs/bioloid_ppo"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment = ExactBioloidGPEnv if args.model == "exact" else BioloidGPEnv
    env_kwargs = (
        {"task": args.task, "goal_distance": args.goal_distance}
        if args.model == "exact"
        else None
    )
    env = make_vec_env(
        environment, n_envs=args.envs, seed=args.seed, env_kwargs=env_kwargs
    )
    if args.resume:
        model = PPO.load(
            args.resume,
            env=env,
            device="auto",
            tensorboard_log=str(args.output.parent / "tensorboard"),
        )
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _: args.learning_rate
        model.clip_range = lambda _: args.clip_range
        model.verbose = 1
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=1024,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=args.clip_range,
            ent_coef=0.005,
            policy_kwargs={"net_arch": [256, 256], "log_std_init": -1.0},
            tensorboard_log=str(args.output.parent / "tensorboard"),
            verbose=1,
            seed=args.seed,
            device="auto",
        )
    model.learn(
        total_timesteps=args.steps,
        progress_bar=False,
        reset_num_timesteps=args.resume is None,
    )
    model.save(args.output)
    env.close()
    print(f"Saved policy to {args.output.with_suffix('.zip')}")


if __name__ == "__main__":
    main()
