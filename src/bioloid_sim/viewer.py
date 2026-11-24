from __future__ import annotations

import time

import mujoco
import mujoco.viewer

from bioloid_sim.env import MODEL_PATH


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = model.key_ctrl[0]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        viewer.cam.distance = 0.85
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15
        while viewer.is_running():
            started = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()
            delay = model.opt.timestep - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)


if __name__ == "__main__":
    main()
