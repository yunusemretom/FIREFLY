"""
PID controller for drone throttle/altitude.
"""

class ThrottlePIDController:
    def __init__(self, kp=0.01, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def compute(self, error):
        return self.kp * error
