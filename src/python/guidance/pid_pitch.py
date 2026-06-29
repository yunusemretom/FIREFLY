"""
PID controller for drone pitch/speed.
"""

# pid_pitch.py
import math
from .pid_core import PID, SlewLimiter, clamp


class PitchController:
    def __init__(self, config):
        op = config["outer_pid"]["speed"]
        self.speed_pid = PID(
            kp=op["kp"],
            ki=op["ki"],
            kd=op["kd"],
            clamp=(op["clamp_min"], op["clamp_max"]),
            integral_limit=op["int_limit"],
        )

        ip = config["inner_pid"]["pitch"]
        self.inner_pid = PID(kp=ip["kp"], ki=ip["ki"], kd=ip["kd"], clamp=(-1.0, 1.0))

        self.slew = SlewLimiter(config["slew"]["pitch_max_change_per_sec"])
        self.camera_tilt = config["flight"]["camera_tilt"]
        self.min_pitch = config["limits"]["min_pitch"]
        self.max_pitch = config["limits"]["max_pitch"]

    def calculate(
        self, state, target_speed_cmd, drone_speed, target_elevation, drone_pitch, yaw_error, dt
    ):
        if state == "KALKIS":
            raw_desired = 0.0
        else:
            speed_error_ms = (target_speed_cmd - drone_speed) / 100.0
            speed_adj = self.speed_pid.calculate(speed_error_ms, dt)

            camera_center_pitch = target_elevation - self.camera_tilt
            raw_desired = camera_center_pitch - speed_adj
            raw_desired = clamp(raw_desired, self.min_pitch, self.max_pitch)

        smooth_desired = self.slew.update(raw_desired, dt)

        # Normalize to [-1, 1] angle command
        if smooth_desired < 0:
            pitch_cmd = smooth_desired / abs(self.min_pitch) if self.min_pitch != 0 else 0.0
        else:
            pitch_cmd = smooth_desired / abs(self.max_pitch) if self.max_pitch != 0 else 0.0

        # Scale pitch by yaw alignment: prevents crabbing sideways before facing the target.
        # cos(0°)=1 (aligned → full pitch), cos(90°)=0 (perpendicular → no pitch).
        yaw_scale = max(0.0, math.cos(math.radians(yaw_error)))
        pitch_cmd = clamp(pitch_cmd * yaw_scale, -1.0, 1.0)

        return smooth_desired, pitch_cmd
