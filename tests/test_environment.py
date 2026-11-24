import numpy as np
import mujoco
from importlib.resources import files

from bioloid_sim import BioloidGPEnv, ExactBioloidGPEnv


def test_model_dimensions_and_finite_rollout():
    env = BioloidGPEnv()
    observation, info = env.reset(seed=123)
    assert observation.shape == (47,)
    assert env.action_space.shape == (18,)
    assert env.model.nu == 18
    assert 0.0 <= info["command_speed"] <= 0.35

    for _ in range(25):
        observation, reward, terminated, truncated, info = env.step(np.zeros(18))
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        if terminated or truncated:
            observation, _ = env.reset(seed=123)
    env.close()


def test_seeded_reset_is_reproducible():
    env = BioloidGPEnv()
    first, first_info = env.reset(seed=9)
    second, second_info = env.reset(seed=9)
    np.testing.assert_allclose(first, second)
    assert first_info == second_info
    env.close()


def test_home_controller_holds_robot_upright():
    env = BioloidGPEnv()
    mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
    env.data.ctrl[:] = env.model.key_ctrl[0]
    for _ in range(1_000):
        mujoco.mj_step(env.model, env.data)

    height = env.data.xpos[env.torso_id, 2]
    upright = env.data.xmat[env.torso_id, 8]
    assert height > 0.19
    assert upright > 0.85
    env.close()


def test_exact_cad_preview_loads():
    path = files("bioloid_sim").joinpath("models/bioloid_gp_exact.xml")
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.nmesh == 1
    assert model.nbody == 2


def test_exact_16dof_model_holds_its_cad_pose():
    path = files("bioloid_sim").joinpath("models/bioloid_gp_step_16dof.xml")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = model.key_ctrl[0]
    mujoco.mj_step(model, data, 1_000)

    assert model.nu == 16
    assert np.isfinite(data.qpos).all()
    assert data.qpos[2] > 0.29
    assert np.linalg.norm(data.qpos[:2]) < 0.01


def test_exact_model_enables_non_adjacent_self_collision_without_home_overlap():
    path = files("bioloid_sim").joinpath("models/bioloid_gp_step_16dof.xml")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    collision_ids = [
        index
        for index in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").endswith(
            "_collision"
        )
    ]
    assert model.nexclude > 0
    assert all(model.geom_conaffinity[index] & 2 for index in collision_ids)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for contact_index in range(data.ncon):
        first = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[contact_index].geom1
        )
        second = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[contact_index].geom2
        )
        assert "floor" in {first, second}


def test_exact_environment_has_16_actions_and_finite_rollout():
    env = ExactBioloidGPEnv()
    observation, info = env.reset(seed=123)
    assert observation.shape == (43,)
    assert env.action_space.shape == (16,)
    assert info["command_speed"] == 0.0

    for _ in range(50):
        observation, reward, terminated, truncated, _ = env.step(np.zeros(16))
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        if terminated or truncated:
            observation, _ = env.reset(seed=123)
    env.close()


def test_exact_environment_random_controls_do_not_destabilize_solver():
    env = ExactBioloidGPEnv(task="walk", goal_distance=0.5)
    observation, _ = env.reset(seed=321)
    rng = np.random.default_rng(321)
    for _ in range(250):
        action = rng.uniform(-0.5, 0.5, 16)
        observation, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        assert not info["unstable"]
        assert info["goal_distance"] == 0.5
        if terminated or truncated:
            observation, _ = env.reset(seed=321)
    env.close()


def test_recovery_mode_starts_fallen_without_terminating_immediately():
    env = ExactBioloidGPEnv(
        task="walk",
        goal_distance=5.0,
        allow_recovery=True,
        fallen_reset_probability=0.0,
    )
    observation, reset_info = env.reset(seed=44, options={"start_fallen": True})
    assert reset_info["start_fallen"]
    assert np.isfinite(observation).all()

    for _ in range(20):
        observation, reward, terminated, truncated, info = env.step(np.zeros(16))
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        assert not info["unstable"]
        assert not terminated
        assert not truncated
    assert info["fallen"]
    env.close()
