"""
nav_integration.py  —  Navigation modülünü drone_gui.py'a bağlayan köprü

"""
try:
    from .ekf import EKF
    from .gnss_filter import GNSSCorruptor, GNSSPrefilter
    from .waypoint_controller import WaypointController
except ImportError:
    from ekf import EKF
    from gnss_filter import GNSSCorruptor, GNSSPrefilter
    from waypoint_controller import WaypointController


class NavigationSystem:
    def __init__(self, corrupt_for_test=True,
                 intercept_time=0.3,  
                 q_vel=2500.0):     
        self.drone_ekf = EKF(q_vel=q_vel)
        self.target_ekf = EKF(q_vel=q_vel)

        self.prefilter = GNSSPrefilter()
        self.corruptor = GNSSCorruptor(enabled=corrupt_for_test)

        self.wp = WaypointController(intercept_time=intercept_time)

    def process(self, telemetry):

        d_pos_raw = tuple(telemetry["drone"]["position"])
        t_pos_raw = tuple(telemetry["target"]["position"])

        d_meas = self.prefilter.prefilter(d_pos_raw)
        self.drone_ekf.step(d_meas)

        t_corrupted = self.corruptor.corrupt(t_pos_raw)   
        t_meas = self.prefilter.prefilter(t_corrupted)
        self.target_ekf.step(t_meas)

        return self.wp.compute(self.drone_ekf, self.target_ekf)