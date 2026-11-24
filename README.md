# Bioloid GP MuJoCo simulator

An 18-DOF Bioloid-style humanoid for learning balance and locomotion with
MuJoCo, Gymnasium, and PPO. The first model intentionally uses primitive
collision geometry: it is fast and stable enough for reinforcement learning.
The supplied `BioloidGP.stp` assembly is the visual-geometry reference; it is
not copied into this repository.

## What works

- free-floating humanoid with 3 joints per arm and 6 joints per leg;
- joint, velocity, and actuator-force limits;
- a neutral standing controller and interactive MuJoCo viewer;
- a Gymnasium environment with randomized resets and a forward-walking reward;
- PPO training and checkpoint evaluation through Stable-Baselines3.

This is a simulation baseline, not yet a calibrated digital twin. Before a
policy is sent to hardware, actuator latency, backlash, real masses, IMU noise,
joint signs, and the AX-12/AX-18 torque curves must be measured and randomized.

## Windows setup

From PowerShell in this directory:

```powershell
& 'C:\Program Files\FreeCAD 1.1\bin\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[train,test]'
```

Open the model with its neutral pose controller:

```powershell
.\.venv\Scripts\bioloid-view.exe
```

Run a short validation training:

```powershell
.\.venv\Scripts\bioloid-train.exe --steps 100000 --envs 4
```

Evaluate the saved policy:

```powershell
.\.venv\Scripts\bioloid-eval.exe runs\bioloid_ppo.zip
```

For a repeatable evaluation without a window:

```powershell
.\.venv\Scripts\bioloid-eval.exe runs\bioloid_ppo.zip --headless --episodes 10
```

Training begins with standing and low forward-speed commands. A useful walking
policy will normally need millions of simulation steps and reward/physics
tuning; the short command above only proves that the complete loop works.

## Model controls

Action values are normalized to `[-1, 1]` and mapped to safe position targets.
The observation contains torso orientation, base velocity, joint offsets from
the home pose, joint velocity, and the commanded forward speed.

Joint order:

```text
r_shoulder_pitch, r_shoulder_roll, r_elbow,
l_shoulder_pitch, l_shoulder_roll, l_elbow,
r_hip_yaw, r_hip_roll, r_hip_pitch, r_knee, r_ankle_pitch, r_ankle_roll,
l_hip_yaw, l_hip_roll, l_hip_pitch, l_knee, l_ankle_pitch, l_ankle_roll
```

## Sources and limitations

Kinematic dimensions and joint naming were cross-checked against the Apache-2.0
[`verlab/bioloid_common`](https://github.com/verlab/bioloid_common) model. The
alternative ROS/Gazebo assembly in
[`fryumbla/ros-bioloid-gp`](https://github.com/fryumbla/ros-bioloid-gp) was used
to compare link masses and the SolidWorks-exported geometry. Both upstream
checkouts are kept under `_sources/community/` and are not imported at runtime.
