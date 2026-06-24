"""
Waypoint controller for FPV drone interception path planning.
"""

import math


class WaypointController:
    def __init__(self, intercept_time=0.3, camera_tilt=25.0):

        self.intercept_time = intercept_time
        self.camera_tilt = camera_tilt

    def compute(self, drone_ekf, target_ekf):

        d_pos = drone_ekf.get_position()
        d_vel = drone_ekf.get_velocity()

        tgt_future = target_ekf.predict_future(self.intercept_time)

        dx = tgt_future[0] - d_pos[0]
        dy = tgt_future[1] - d_pos[1]
        dz = tgt_future[2] - d_pos[2]

        dist_2d = math.sqrt(dx * dx + dy * dy)
        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

        target_yaw = math.degrees(math.atan2(dy, dx))
        target_pitch_world = math.degrees(math.atan2(dz, dist_2d if dist_2d > 1e-6 else 1e-6))

        return {
            # Ham waypoint (filtrelenmiş hedef konum verişlmiştir burada) Kerem'in UI'ına buradan gidecektir, konuşulabilir
            "waypoint": tgt_future,
            "drone_pos": d_pos,
            "dx": dx, "dy": dy, "dz": dz,
            "dist_2d": dist_2d,
            "dist_3d": dist_3d,
            # PID girdileri (Meryem'den gelecek sabitler)
            "target_yaw": target_yaw,           
            "target_pitch_world": target_pitch_world,  
            "target_speed": target_ekf.get_speed(),
            "drone_speed": drone_ekf.get_speed(),
        }