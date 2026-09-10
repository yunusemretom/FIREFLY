# Firefly

Autonomous pursuit controller for a kamikaze drone: it connects to an Unreal
Engine simulator over TCP, estimates a maneuvering aerial target's state with an
extended Kalman filter, and holds a trailing intercept position with cascaded
PID control.

Built for the TEKNOFEST "Drone of War" competition.

![demo](docs/demo.gif)

## Tech stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-EKF-013243?logo=numpy&logoColor=white)
![Tkinter](https://img.shields.io/badge/Tkinter-GCS-FFD43B)
![OpenCV](https://img.shields.io/badge/OpenCV-Vision-5C3EE8?logo=opencv&logoColor=white)
![Unreal Engine](https://img.shields.io/badge/Unreal_Engine-Simulator-0E1128?logo=unrealengine&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

## Quick start

The simulator must be running and listening on `localhost:12345` before you
start the ground station.

```bash
# 1. Clone
git clone https://github.com/yunusemretom/FIREFLY.git
cd FIREFLY

# 2. Virtual environment and dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Start the simulator (separate window), then launch the ground station
python drone_gui.py
```

Use the SDK directly if you want to script a run instead of flying it from the
interface:

```python
import drone_sdk as drone

drone.connect()                                   # TCP to localhost:12345
drone.set_control_surfaces(thr, pitch, roll, yaw, True)
telemetry = drone.get_telemetry()                 # position, rotation, speed
```

Gains and limits are in `config/`. The control loop runs at 20 Hz on a 50 ms
period.

## How it works

Two components: a **drone SDK** that owns the socket and the telemetry parsing,
and a **ground station** that owns the control loop, the mode logic and the
operator interface. Guidance, navigation and simulation code sits under
`src/python/`, grouped by responsibility.

```
comms/       socket, telemetry parsing, data broker
navigation/  EKF, GNSS filtering, waypoint control
guidance/    PID core, pursuit controller, mode manager
vision/      detector, integrated vision pipeline
simulation/  scenario runner, GNSS jammer, sim bridge
```

### Estimating where the target will be

Position measurements arrive noisy and irregular, and the naive approach of
differencing consecutive positions to get velocity amplifies that noise until
the derivative term is useless.

`navigation/ekf.py` runs a six-state constant-acceleration filter over position
and velocity. Three details in it exist because of specific failures:

- **Chi-squared gating.** A measurement whose Mahalanobis distance exceeds
  `chi2_threshold` is rejected rather than fused. A single bad fix used to snap
  the estimate hard enough to command a full-authority control spike.
- **Adaptive measurement covariance.** Consecutive rejections grow `R` by
  `r_scale_growth`, up to `r_scale_max`, and it decays back by `r_scale_decay`
  when measurements start agreeing again. Without this the filter would gate out
  a genuine target maneuver forever, having decided reality was the outlier.
- **A rejection-streak reset.** After `_reject_reset` consecutive rejections the
  filter reinitializes from the buffer instead of defending a diverged estimate.

### Not overshooting the target

The pursuit geometry is the same problem a fixed-wing has, in a form where you
can hover. `guidance/pursuit.py` computes a virtual follow point
`follow_distance_cm` behind the target along its velocity vector, then regulates
the **signed** distance to it:

```
s = (drone_position - follow_point) . target_heading

s > 0  too close, about to overshoot  -> slow down
s = 0  on station                     -> match target speed
s < 0  falling behind                 -> speed up
```

Feeding `-s` into a PID makes zero the correct answer, so the same controller
handles closing and lagging without a mode switch:

```
target_speed = target_speed + dist_pid.calculate(-s)
```

### The spin bug

This one was not obvious from the math. When the drone sat exactly on station,
the follow point occasionally landed slightly *behind* it. Yaw was pointed at the
follow point, so the drone commanded a 180 degree turn, lost the target, and
spun.

The fix separates the two uses of the geometry: **yaw always points at the
target, never at the follow point.** Station-keeping is handled purely by
throttling forward speed and letting the target open the gap on its own. A
control loop should not need to turn around to correct a distance error.

`SlewLimiter` in `guidance/pid_core.py` caps how fast any command can change per
second, which stopped the actuator saturation that mode transitions used to
cause.

## Known limitations

- Simulator only. No flight on real hardware, so aerodynamic and latency effects
  are unmodeled.
- PID gains are tuned for the competition airframe in the simulator and do not
  transfer as-is.
- The EKF assumes constant acceleration and lags a hard, sustained maneuver.
- Vision runs as a separate pipeline and is not yet fused into the EKF; the
  navigation solution is position-measurement driven.
- Single target. No track association or multi-target logic.
- The interface is Tkinter, which is adequate for a competition bench but not
  for field use.

## Roadmap

- Fuse the vision bearing into the EKF measurement model.
- Replace the fixed process noise with an adaptive or IMM filter so hard
  maneuvers are tracked without loosening the gate.
- Port the control loop to a real airframe over MAVLink.
- Record scenario runs and score them automatically instead of judging by eye.

## License

MIT. See [LICENSE](LICENSE).

Turkish documentation is preserved in [README.tr.md](README.tr.md).
