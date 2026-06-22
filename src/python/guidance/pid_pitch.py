"""
PID controller for drone pitch/elevation.
"""

# pid_pitch.py
from .pid_core import PID, SlewLimiter, clamp
class PitchController:
    def __init__(self, config):
        op = config['outer_pid']['speed']
        self.speed_pid = PID(kp=op['kp'], ki=op['ki'], kd=op['kd'], 
                             clamp=(op['clamp_min'], op['clamp_max']), 
                             integral_limit=op['int_limit'])
        
        ip = config['inner_pid']['pitch']
        self.inner_pid = PID(kp=ip['kp'], ki=ip['ki'], kd=ip['kd'], clamp=(-1.0, 1.0))
        
        self.slew = SlewLimiter(config['slew']['pitch_max_change_per_sec'])
        self.camera_tilt = config['flight']['camera_tilt']
        self.min_pitch = config['limits']['min_pitch']
        self.max_pitch = config['limits']['max_pitch']
    def calculate(self, state, target_speed_cmd, drone_speed, target_elevation, drone_pitch, dt):
        if state == "KALKIS":
            raw_desired = -10.0
        else:
            speed_error_ms = (target_speed_cmd - drone_speed) / 100.0
            speed_adj = self.speed_pid.calculate(speed_error_ms, dt)
            
            camera_center_pitch = target_elevation - self.camera_tilt
            raw_desired = camera_center_pitch - speed_adj
            raw_desired = clamp(raw_desired, self.min_pitch, self.max_pitch)
        smooth_desired = self.slew.update(raw_desired, dt)
        pitch_error = smooth_desired - drone_pitch
        pitch_cmd = self.inner_pid.calculate(pitch_error, dt)
        
        return smooth_desired, pitch_cmd