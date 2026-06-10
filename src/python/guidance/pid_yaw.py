"""
PID controller for drone yaw/heading.
"""

class YawPIDController:
    def __init__(self, kp=0.07, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def compute(self, error):
        return self.kp * error
