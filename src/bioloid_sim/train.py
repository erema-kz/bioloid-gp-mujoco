from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from bioloid_sim.env import BioloidGPEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Bioloid GP with PPO")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("runs/bioloid_ppo"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(BioloidGPEnv, n_envs=args.envs, seed=args.seed)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs={"net_arch": [256, 256], "log_std_init": -1.0},
        tensorboard_log=str(args.output.parent / "tensorboard"),
        verbose=1,
        seed=args.seed,
        device="auto",
    )
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(args.output)
    env.close()
    print(f"Saved policy to {args.output.with_suffix('.zip')}")


if __name__ == "__main__":
    main()
