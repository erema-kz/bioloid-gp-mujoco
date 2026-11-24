from __future__ import annotations

from importlib.resources import files
import time

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


EXACT_STEP_MODEL_PATH = files("bioloid_sim").joinpath("models/bioloid_gp_step_16dof.xml")


class ExactBioloidGPEnv(gym.Env[np.ndarray, np.ndarray]):
    """Balance or forward-walking task for the user's exact 16-DOF STEP robot."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        episode_seconds: float = 12.0,
        task: str = "balance",
        goal_distance: float = 0.5,
    ):
        super().__init__()
        if task not in {"balance", "walk"}:
            raise ValueError(f"Unknown task: {task}")
        self.task = task
        self.goal_distance = goal_distance
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(str(EXACT_STEP_MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.control_dt = 0.02
        self.frame_skip = round(self.control_dt / self.model.opt.timestep)
        self.max_steps = int(episode_seconds / (self.model.opt.timestep * self.frame_skip))
        self._step = 0
        self._viewer = None
        self._renderer = None
        self._last_render_time = None

        self.joint_qpos_ids = np.array(
            [self.model.jnt_qposadr[i] for i in range(1, self.model.njnt)], dtype=np.int32
        )
        self.joint_qvel_ids = np.array(
            [self.model.jnt_dofadr[i] for i in range(1, self.model.njnt)], dtype=np.int32
        )
        self.home = self.model.key_qpos[0, self.joint_qpos_ids].copy()
        self.ctrl_center = self.home.copy()
        self.ctrl_half_range = 0.22 * (
            self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]
        )
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.command_speed = 0.0
        self._filtered_action = np.zeros(16, dtype=np.float64)
        self._previous_action = np.zeros(16, dtype=np.float64)
        self._start_forward_position = 0.0
        self._previous_distance = 0.0

        observation_size = 4 + 6 + 16 + 16 + 1
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(observation_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32)

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
        self.data.qpos[self.joint_qpos_ids] += self.np_random.uniform(-0.015, 0.015, 16)
        self.data.qvel[:] = self.np_random.normal(0.0, 0.005, self.model.nv)
        self.command_speed = (
            float(self.np_random.uniform(0.06, 0.22)) if self.task == "walk" else 0.0
        )
        self._filtered_action.fill(0.0)
        self._previous_action.fill(0.0)
        self._start_forward_position = float(-self.data.qpos[1])
        self._previous_distance = 0.0
        self._last_render_time = None
        self._step = 0
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), {"command_speed": self.command_speed}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._filtered_action = 0.82 * self._filtered_action + 0.18 * action
        target = self.ctrl_center + self._filtered_action * self.ctrl_half_range
        self.data.ctrl[:] = np.clip(
            target, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        )
        mujoco.mj_step(self.model, self.data, self.frame_skip)
        self._step += 1

        height = float(self.data.xpos[self.base_id, 2])
        # The CAD model is Y-up; the root's +Y axis is world +Z in its home pose.
        upright = float(self.data.xmat[self.base_id, 7])
        forward_velocity = float(-self.data.qvel[1])
        effort = float(np.mean(np.square(self.data.actuator_force / 2.0)))
        action_cost = float(np.mean(np.square(self._filtered_action)))
        action_rate = float(np.mean(np.square(self._filtered_action - self._previous_action)))
        self._previous_action[:] = self._filtered_action
        distance = float(-self.data.qpos[1] - self._start_forward_position)
        progress = distance - self._previous_distance
        self._previous_distance = distance
        success = bool(self.task == "walk" and distance >= self.goal_distance)
        unstable = bool(np.any(self.data.warning.number > 0))

        if self.task == "walk":
            tracking = np.exp(-((forward_velocity - self.command_speed) / 0.08) ** 2)
            # Standing still used to be a strong local optimum: the upright/alive
            # reward was larger than the small per-step displacement reward.  Make
            # commanded forward motion dominate while still penalising a fall.
            directed_velocity = float(np.clip(forward_velocity, -0.20, 0.30))
            stopped_penalty = 0.20 if forward_velocity < 0.02 else 0.0
            reward = (
                0.10 * max(upright, 0.0)
                + 2.0 * tracking
                + 4.0 * directed_velocity
                - 0.35 * self.data.qvel[0] ** 2
                - 0.12 * self.data.qvel[5] ** 2
                - 0.012 * effort
                - 0.01 * action_cost
                - 0.03 * action_rate
                - stopped_penalty
                + 100.0 * progress
            )
        else:
            reward = (
                1.0
                + 1.5 * max(upright, 0.0)
                - 0.25 * np.sum(np.square(self.data.qvel[:3]))
                - 0.08 * np.sum(np.square(self.data.qvel[3:6]))
                - 0.01 * effort
                - 0.01 * action_cost
                - 0.03 * action_rate
            )

        terminated = bool(
            height < 0.20
            or height > 0.42
            or upright < 0.45
            or unstable
            or not np.isfinite(self.data.qpos).all()
            or success
        )
        if success:
            reward += 100.0
        elif terminated:
            reward -= 25.0
        truncated = self._step >= self.max_steps
        info = {
            "height": height,
            "upright": upright,
            "forward_velocity": forward_velocity,
            "command_speed": self.command_speed,
            "distance": distance,
            "goal_distance": self.goal_distance,
            "success": success,
            "unstable": unstable,
        }
        if self.render_mode == "human":
            self.render()
        return self._observation(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer as mujoco_viewer

                self._viewer = mujoco_viewer.launch_passive(self.model, self.data)
            if self._last_render_time is None:
                self._last_render_time = time.perf_counter()
            target_time = self._last_render_time + self.control_dt
            remaining = target_time - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
                self._last_render_time = target_time
            else:
                self._last_render_time = time.perf_counter()
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
