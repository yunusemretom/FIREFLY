"""
PID controller for drone throttle/altitude.
"""

# pid_throttle.py
import math
from .pid_core import PID, clamp

class ThrottleController:
    def __init__(self, config):
        self.base_throttle = config['flight']['base_throttle']
        op = config['outer_pid']['altitude']
        self.alt_pid = PID(kp=op['kp'], ki=op['ki'], kd=op['kd'], 
                           clamp=(op['clamp_min'], op['clamp_max']), 
                           integral_limit=op['int_limit'])
        
        self.min_t = config['limits']['min_throttle']
        self.max_t = config['limits']['max_throttle']

    def calculate(self, dz, current_pitch, dt):
        dz_meters = clamp(dz / 100.0, -5.0, 5.0)
        alt_correction = self.alt_pid.calculate(dz_meters, dt)
        
        # O meşhur mükemmel geometrik denklemimiz!
        current_pitch_rad = math.radians(clamp(abs(current_pitch), 0.0, 60.0))
        target_throttle = (self.base_throttle + alt_correction) / max(0.5, math.cos(current_pitch_rad))
        
        return clamp(target_throttle, self.min_t, self.max_t)