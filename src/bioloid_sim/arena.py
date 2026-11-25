from __future__ import annotations

import argparse
from copy import deepcopy
from importlib.resources import files
import json
from pathlib import Path
import math
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from stable_baselines3 import PPO

from bioloid_sim.hybrid import nearest_fall_direction, reference_fall_quaternions


MODEL_PATH = files("bioloid_sim").joinpath("models/bioloid_gp_step_16dof.xml")
JOINT_ORDER = (
    "right_arm_1",
    "right_arm_2",
    "right_arm_3",
    "left_arm_1",
    "left_arm_2",
    "left_arm_3",
    "right_leg_1",
    "right_leg_2",
    "right_leg_3",
    "right_leg_4",
    "right_leg_5",
    "left_leg_1",
    "left_leg_2",
    "left_leg_3",
    "left_leg_4",
    "left_leg_5",
)
REFERENCE_ATTRIBUTES = {"joint", "site", "body1", "body2", "objname"}


def _prefix_tree(element: ET.Element, prefix: str) -> None:
    for node in element.iter():
        if "name" in node.attrib:
            node.attrib["name"] = prefix + node.attrib["name"]
        for attribute in REFERENCE_ATTRIBUTES:
            if attribute in node.attrib:
                node.attrib[attribute] = prefix + node.attrib[attribute]


def _grid_position(index: int, count: int, spacing: float = 0.48) -> tuple[float, float]:
    columns = min(4, math.ceil(math.sqrt(count)))
    rows = math.ceil(count / columns)
    column = index % columns
    row = index // columns
    return (
        (column - (columns - 1) / 2) * spacing,
        (row - (rows - 1) / 2) * spacing,
    )


def build_arena_xml(robot_count: int) -> str:
    if not 1 <= robot_count <= 12:
        raise ValueError("robot_count must be between 1 and 12")
    tree = ET.parse(str(MODEL_PATH))
    root = tree.getroot()
    compiler = root.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", str(Path(str(MODEL_PATH)).parent))

    world = root.find("worldbody")
    actuator = root.find("actuator")
    contact = root.find("contact")
    assert world is not None and actuator is not None and contact is not None
    source_body = world.find("body[@name='base']")
    assert source_body is not None
    source_actuators = [deepcopy(node) for node in actuator]
    source_excludes = [deepcopy(node) for node in contact]
    world.remove(source_body)
    actuator.clear()
    contact.clear()

    sensor = root.find("sensor")
    if sensor is not None:
        root.remove(sensor)
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)

    for index in range(robot_count):
        prefix = f"robot{index}_"
        x, y = _grid_position(index, robot_count)
        body = deepcopy(source_body)
        _prefix_tree(body, prefix)
        body.set("pos", f"{x:.6f} {y:.6f} 0.295034")
        world.append(body)
        for source in source_actuators:
            node = deepcopy(source)
            _prefix_tree(node, prefix)
            actuator.append(node)
        for source in source_excludes:
            node = deepcopy(source)
            _prefix_tree(node, prefix)
            contact.append(node)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


class RobotController:
    def __init__(self, model: mujoco.MjModel, index: int):
        prefix = f"robot{index}_"
        root_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "root"
        )
        self.root_qpos = int(model.jnt_qposadr[root_joint])
        self.root_qvel = int(model.jnt_dofadr[root_joint])
        self.base_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, prefix + "base"
        )
        joint_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name + "_joint"
                )
                for name in JOINT_ORDER
            ],
            dtype=np.int32,
        )
        self.joint_qpos = model.jnt_qposadr[joint_ids].astype(np.int32)
        self.joint_qvel = model.jnt_dofadr[joint_ids].astype(np.int32)
        self.actuators = np.asarray(
            [
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name + "_act"
                )
                for name in JOINT_ORDER
            ],
            dtype=np.int32,
        )
        ranges = model.actuator_ctrlrange[self.actuators]
        self.ctrl_center = np.mean(ranges, axis=1)
        self.ctrl_half_range = 0.22 * (ranges[:, 1] - ranges[:, 0])
        self.filtered_action = np.zeros(16, dtype=np.float64)
        self.command_speed = 0.14 + 0.015 * (index % 3)
        self.mode = "walk"
        self.fallen_steps = 0
        self.ready_steps = 0
        self.trajectory_step = 0
        self.direction = "front"

    def observation(self, data: mujoco.MjData) -> np.ndarray:
        return np.concatenate(
            (
                data.qpos[self.root_qpos + 3 : self.root_qpos + 7],
                np.clip(data.qvel[self.root_qvel : self.root_qvel + 6], -10.0, 10.0),
                data.qpos[self.joint_qpos],
                np.clip(data.qvel[self.joint_qvel], -20.0, 20.0),
                np.asarray([self.command_speed]),
            )
        ).astype(np.float32)

    def apply_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        action: np.ndarray,
        recovery: bool = False,
    ) -> None:
        self.filtered_action = 0.82 * self.filtered_action + 0.18 * np.clip(
            action, -1.0, 1.0
        )
        ranges = model.actuator_ctrlrange[self.actuators]
        target_range = self.ctrl_half_range
        if recovery:
            upright = float(data.xmat[self.base_id, 7])
            recovery_blend = 1.0 - float(np.clip(upright / 0.70, 0.0, 1.0))
            target_range = (0.22 + 0.26 * recovery_blend) * (
                ranges[:, 1] - ranges[:, 0]
            )
        targets = self.ctrl_center + self.filtered_action * target_range
        data.ctrl[self.actuators] = np.clip(targets, ranges[:, 0], ranges[:, 1])

    def update_recovery_mode(
        self,
        data: mujoco.MjData,
        references: dict[str, np.ndarray],
        available_directions: set[str],
    ) -> None:
        height = float(data.xpos[self.base_id, 2])
        upright = float(data.xmat[self.base_id, 7])
        ready = height > 0.20 and upright > 0.48
        fallen = height < 0.19 or upright < 0.35
        if self.mode == "getup":
            self.ready_steps = self.ready_steps + 1 if ready else 0
            if self.ready_steps >= 5:
                self.mode = "walk"
                self.fallen_steps = 0
        else:
            self.fallen_steps = self.fallen_steps + 1 if fallen else 0
            if self.fallen_steps >= 8:
                self.mode = "getup"
                self.direction = nearest_fall_direction(
                    data.qpos[self.root_qpos + 3 : self.root_qpos + 7],
                    references,
                    available_directions,
                )
                self.trajectory_step = 0
                self.ready_steps = 0

    def recovery_action(
        self, trajectories: dict[str, np.ndarray], steps_per_keyframe: int
    ) -> np.ndarray:
        keyframes = trajectories[self.direction]
        phase = self.trajectory_step / steps_per_keyframe
        low = min(int(phase), len(keyframes) - 1)
        high = min(low + 1, len(keyframes) - 1)
        blend = phase - int(phase)
        action = (1.0 - blend) * keyframes[low] + blend * keyframes[high]
        self.trajectory_step = (self.trajectory_step + 1) % (
            len(keyframes) * steps_per_keyframe
        )
        return action


def choose_settings() -> tuple[int, float, float, float, bool, float] | None:
    import tkinter as tk
    from tkinter import ttk

    result: list[tuple[int, float, float, float, bool, float]] = []
    root = tk.Tk()
    root.title("Bioloid GP Arena")
    root.resizable(False, False)
    frame = ttk.Frame(root, padding=18)
    frame.grid()
    robots = tk.IntVar(value=4)
    speed = tk.DoubleVar(value=1.0)
    push_enabled = tk.BooleanVar(value=True)
    push_interval = tk.DoubleVar(value=6.0)
    push_force = tk.DoubleVar(value=35.0)
    auto_recover = tk.BooleanVar(value=True)
    lifetime = tk.DoubleVar(value=7.0)
    ttk.Label(frame, text="Number of robots").grid(row=0, column=0, sticky="w")
    ttk.Spinbox(frame, from_=1, to=12, textvariable=robots, width=8).grid(
        row=0, column=1, padx=(12, 0)
    )
    ttk.Label(frame, text="Simulation speed").grid(
        row=1, column=0, sticky="w", pady=(12, 0)
    )
    speed_label = ttk.Label(frame, text="1.0×", width=8)
    speed_label.grid(row=1, column=1, padx=(12, 0), pady=(12, 0))

    def update_speed(value: str) -> None:
        speed_label.configure(text=f"{float(value):.1f}×")

    ttk.Scale(
        frame,
        from_=0.25,
        to=8.0,
        variable=speed,
        command=update_speed,
        length=280,
    ).grid(row=2, column=0, columnspan=2, pady=(6, 16))

    ttk.Checkbutton(
        frame, text="Random physical pushes", variable=push_enabled
    ).grid(row=3, column=0, columnspan=2, sticky="w")
    ttk.Label(frame, text="Push interval, seconds").grid(
        row=4, column=0, sticky="w", pady=(10, 0)
    )
    ttk.Spinbox(
        frame, from_=1.0, to=60.0, increment=0.5, textvariable=push_interval, width=8
    ).grid(row=4, column=1, padx=(12, 0), pady=(10, 0))
    ttk.Label(frame, text="Push force, N").grid(
        row=5, column=0, sticky="w", pady=(10, 0)
    )
    ttk.Spinbox(
        frame, from_=5.0, to=100.0, increment=5.0, textvariable=push_force, width=8
    ).grid(row=5, column=1, padx=(12, 0), pady=(10, 0))
    ttk.Checkbutton(
        frame, text="Automatic get-up after a fall", variable=auto_recover
    ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(12, 12))
    ttk.Label(frame, text="Lifetime, seconds").grid(row=7, column=0, sticky="w")
    ttk.Spinbox(
        frame, from_=1.0, to=120.0, increment=1.0, textvariable=lifetime, width=8
    ).grid(row=7, column=1, padx=(12, 0))
    ttk.Label(
        frame,
        text="Each push acts on a random robot for 0.25 seconds",
        foreground="#666666",
    ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(12, 14))

    def start() -> None:
        interval = max(0.5, push_interval.get()) if push_enabled.get() else 0.0
        result.append(
            (
                max(1, min(12, robots.get())),
                max(0.25, speed.get()),
                interval,
                max(0.0, push_force.get()),
                auto_recover.get(),
                max(1.0, lifetime.get()),
            )
        )
        root.destroy()

    ttk.Button(frame, text="Launch", command=start).grid(
        row=9, column=0, columnspan=2
    )
    root.mainloop()
    return result[0] if result else None


def run_arena(
    checkpoint: Path,
    robot_count: int,
    speed: float,
    headless: bool = False,
    seconds: float = 0.0,
    trajectories_path: Path = Path("runs/getup_trajectories.json"),
    push_interval: float = 0.0,
    push_force: float = 35.0,
    auto_recover: bool = False,
    lifetime: float = 0.0,
) -> None:
    model = mujoco.MjModel.from_xml_string(build_arena_xml(robot_count))
    data = mujoco.MjData(model)
    controllers = [RobotController(model, index) for index in range(robot_count)]
    for index, controller in enumerate(controllers):
        x, y = _grid_position(index, robot_count)
        data.qpos[controller.root_qpos : controller.root_qpos + 7] = (
            x,
            y,
            0.295034,
            0.707106781,
            0.707106781,
            0.0,
            0.0,
        )
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    initial_forward = np.asarray(
        [-data.qpos[controller.root_qpos + 1] for controller in controllers]
    )
    policy = PPO.load(checkpoint)
    trajectories: dict[str, np.ndarray] = {}
    steps_per_keyframe = 1
    references: dict[str, np.ndarray] = {}
    if auto_recover:
        trajectory_data = json.loads(trajectories_path.read_text(encoding="utf-8"))
        steps_per_keyframe = int(trajectory_data["steps_per_keyframe"])
        trajectories = {
            name: np.asarray(value["keyframes"], dtype=np.float64)
            for name, value in trajectory_data["directions"].items()
        }
        if not trajectories:
            raise ValueError("The trajectory file contains no directions")
        references = reference_fall_quaternions(
            np.asarray((0.707106781, 0.707106781, 0.0, 0.0))
        )
    control_dt = 0.02
    frame_skip = round(control_dt / model.opt.timestep)
    simulated_seconds = 0.0
    rng = np.random.default_rng(1234)
    next_push = push_interval if push_interval > 0.0 else math.inf
    push_until = -1.0
    pushed_controller: RobotController | None = None
    push_vector = np.zeros(3, dtype=np.float64)
    next_reset = lifetime if lifetime > 0.0 else math.inf

    viewer = None
    if not headless:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = (0.0, 0.0, 0.20)
        viewer.cam.distance = 0.75 + 0.30 * math.sqrt(robot_count)
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -22

    try:
        while viewer is None or viewer.is_running():
            loop_started = time.perf_counter()
            if auto_recover:
                for controller in controllers:
                    controller.update_recovery_mode(
                        data, references, set(trajectories)
                    )
            observations = np.stack(
                [controller.observation(data) for controller in controllers]
            )
            actions, _ = policy.predict(observations, deterministic=True)
            for controller, action in zip(controllers, actions):
                recovery = auto_recover and controller.mode == "getup"
                if recovery:
                    action = controller.recovery_action(
                        trajectories, steps_per_keyframe
                    )
                controller.apply_action(model, data, action, recovery=recovery)

            data.xfrc_applied[:] = 0.0
            if simulated_seconds >= next_push:
                pushed_controller = controllers[int(rng.integers(robot_count))]
                angle = float(rng.uniform(0.0, 2.0 * math.pi))
                push_vector[:] = (
                    push_force * math.cos(angle),
                    push_force * math.sin(angle),
                    0.0,
                )
                push_until = simulated_seconds + 0.25
                next_push += push_interval
            if pushed_controller is not None and simulated_seconds < push_until:
                data.xfrc_applied[pushed_controller.base_id, :3] = push_vector
            mujoco.mj_step(model, data, frame_skip)
            simulated_seconds += control_dt
            if simulated_seconds >= next_reset:
                data.qpos[:] = initial_qpos
                data.qvel[:] = 0.0
                data.ctrl[:] = 0.0
                data.xfrc_applied[:] = 0.0
                for controller in controllers:
                    controller.filtered_action.fill(0.0)
                    controller.mode = "walk"
                    controller.fallen_steps = 0
                    controller.ready_steps = 0
                    controller.trajectory_step = 0
                pushed_controller = None
                push_until = -1.0
                mujoco.mj_forward(model, data)
                next_reset += lifetime
            if viewer is not None:
                viewer.sync()
                remaining = control_dt / speed - (time.perf_counter() - loop_started)
                if remaining > 0:
                    time.sleep(remaining)
            if seconds > 0 and simulated_seconds >= seconds:
                break
    finally:
        if viewer is not None:
            viewer.close()

    distances = np.asarray(
        [-data.qpos[controller.root_qpos + 1] for controller in controllers]
    ) - initial_forward
    print(
        f"robots={robot_count} speed={speed:.2f}x "
        f"lifetime={lifetime:.1f}s push_interval={push_interval:.1f}s "
        f"auto_recover={auto_recover} "
        f"mean_forward_position={np.mean(distances):.3f} m"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch multiple Bioloid GP robots")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/bioloid_exact_walk_0p5m_v2.zip")
    )
    parser.add_argument("--robots", type=int)
    parser.add_argument("--speed", type=float)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument(
        "--trajectories", type=Path, default=Path("runs/getup_trajectories.json")
    )
    parser.add_argument("--push-interval", type=float, default=0.0)
    parser.add_argument("--push-force", type=float, default=35.0)
    parser.add_argument("--auto-recover", action="store_true")
    parser.add_argument(
        "--lifetime", type=float, default=7.0, help="seconds before arena reset"
    )
    args = parser.parse_args()

    if args.robots is None and not args.headless:
        settings = choose_settings()
        if settings is None:
            return
        robot_count, speed, push_interval, push_force, auto_recover, lifetime = settings
    else:
        robot_count = args.robots or 1
        speed = args.speed or 1.0
        push_interval = max(0.0, args.push_interval)
        push_force = max(0.0, args.push_force)
        auto_recover = args.auto_recover
        lifetime = max(0.0, args.lifetime)
    run_arena(
        args.checkpoint,
        robot_count,
        speed,
        args.headless,
        args.seconds,
        args.trajectories,
        push_interval,
        push_force,
        auto_recover,
        lifetime,
    )


if __name__ == "__main__":
    main()
