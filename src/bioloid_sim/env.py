from __future__ import annotations

from importlib.resources import files

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


MODEL_PATH = files("bioloid_sim").joinpath("models/bioloid_gp.xml")


class BioloidGPEnv(gym.Env[np.ndarray, np.ndarray]):
    """Balance and commanded-forward-velocity task for Bioloid GP."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, episode_seconds: float = 12.0):
        super().__init__()
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 10
        self.max_steps = int(episode_seconds / (self.model.opt.timestep * self.frame_skip))
        self._step = 0
        self._viewer = None
        self._renderer = None

        self.joint_qpos_ids = np.array(
            [self.model.jnt_qposadr[i] for i in range(1, self.model.njnt)], dtype=np.int32
        )
        self.joint_qvel_ids = np.array(
            [self.model.jnt_dofadr[i] for i in range(1, self.model.njnt)], dtype=np.int32
        )
        self.home = self.model.key_qpos[0, self.joint_qpos_ids].copy()
        self.ctrl_center = self.home.copy()
        # Keep the first curriculum around the stable crouch instead of exposing
        # every mechanical stop to an untrained policy on its first rollout.
        self.ctrl_half_range = 0.25 * (
            self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]
        )
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        obs_size = 4 + 6 + 18 + 18 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        self.command_speed = 0.0

    def _observation(self) -> np.ndarray:
        return np.concatenate(
            (
                self.data.qpos[3:7],
                np.clip(self.data.qvel[:6], -10.0, 10.0),
                self.data.qpos[self.joint_qpos_ids] - self.home,
                np.clip(self.data.qvel[self.joint_qvel_ids], -20.0, 20.0),
                np.array([self.command_speed]),
            )
        ).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[self.joint_qpos_ids] += self.np_random.uniform(-0.025, 0.025, 18)
        self.data.qvel[:] = self.np_random.normal(0.0, 0.01, self.model.nv)
        self.command_speed = float(self.np_random.uniform(0.0, 0.35))
        self._step = 0
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), {"command_speed": self.command_speed}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = self.ctrl_center + action * self.ctrl_half_range
        self.data.ctrl[:] = np.clip(
            target, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        )
        mujoco.mj_step(self.model, self.data, self.frame_skip)
        self._step += 1

        height = float(self.data.xpos[self.torso_id, 2])
        upright = float(self.data.xmat[self.torso_id, 8])
        forward_velocity = float(self.data.qvel[0])
        tracking = np.exp(-4.0 * (forward_velocity - self.command_speed) ** 2)
        effort = float(np.mean(np.square(self.data.actuator_force / 2.0)))
        action_rate = float(np.mean(np.square(action)))
        reward = 0.6 + 1.5 * tracking + 0.8 * max(upright, 0.0) - 0.015 * effort - 0.01 * action_rate

        terminated = bool(height < 0.15 or height > 0.38 or upright < 0.35 or not np.isfinite(self.data.qpos).all())
        truncated = self._step >= self.max_steps
        info = {
            "height": height,
            "upright": upright,
            "forward_velocity": forward_velocity,
            "command_speed": self.command_speed,
        }
        if self.render_mode == "human":
            self.render()
        return self._observation(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera="tracking")
            return self._renderer.render()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
