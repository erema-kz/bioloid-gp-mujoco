import numpy as np
import mujoco

from bioloid_sim import BioloidGPEnv


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
