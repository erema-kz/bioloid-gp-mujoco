from __future__ import annotations

import argparse
import time
from importlib.resources import files

import mujoco
import mujoco.viewer

from bioloid_sim.env import MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a Bioloid GP model in MuJoCo")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--exact", action="store_true", help="run the articulated exact 16-DOF model")
    choice.add_argument("--cad", action="store_true", help="show the monolithic static CAD assembly")
    args = parser.parse_args()

    if args.exact:
        model_path = files("bioloid_sim").joinpath("models/bioloid_gp_step_16dof.xml")
    elif args.cad:
        model_path = files("bioloid_sim").joinpath("models/bioloid_gp_exact.xml")
    else:
        model_path = MODEL_PATH
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.ctrl[:] = model.key_ctrl[0]
    else:
        mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        if args.cad:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, "preview"
            )
        else:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                "base_link" if args.exact else "torso",
            )
            viewer.cam.distance = 0.55 if args.exact else 0.85
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
