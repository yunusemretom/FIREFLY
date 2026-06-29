import math
from .pid_core import PID, wrap_180, clamp


class YawController:
    def __init__(self, config):
        ip = config["inner_pid"]["yaw"]
        self.inner_pid = PID(kp=ip["kp"], ki=ip["ki"], kd=ip["kd"], clamp=(-1.0, 1.0))

    def calculate(self, dx, dy, drone_yaw, dt):
        target_bearing = math.degrees(math.atan2(dy, dx))
        yaw_error = wrap_180(target_bearing - drone_yaw)
        yaw_cmd = clamp(self.inner_pid.calculate(yaw_error, dt), -1.0, 1.0)

        return yaw_error, yaw_cmd

