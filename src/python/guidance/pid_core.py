# pid_core.py
class PID:
    def __init__(self, kp, ki, kd, clamp=None, integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.clamp = clamp
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_run = True

    def calculate(self, error, dt):
        if self.first_run:
            self.prev_error = error
            self.first_run = False
        self.integral += error * dt
        if self.integral_limit:
            self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        if self.clamp:
            output = max(self.clamp[0], min(self.clamp[1], output))
        return output

class SlewLimiter:
    def __init__(self, max_change_per_sec):
        self.max_change = max_change_per_sec
        self.current_val = 0.0
        self.initialized = False

    def update(self, target_val, dt):
        if not self.initialized:
            self.current_val = target_val
            self.initialized = True
            return self.current_val
        delta = target_val - self.current_val
        max_delta = self.max_change * dt
        self.current_val += max(-max_delta, min(max_delta, delta))
        return self.current_val

def wrap_180(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

def clamp(val, lo, hi):
    return max(lo, min(hi, val))