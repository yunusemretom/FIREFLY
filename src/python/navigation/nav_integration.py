
import time

try:
    from .ekf import EKF
    from .gnss_filter import GNSSCorruptor, GNSSPrefilter
    from .waypoint_controller import WaypointController
except ImportError:
    from ekf import EKF
    from gnss_filter import GNSSCorruptor, GNSSPrefilter
    from waypoint_controller import WaypointController


class NavigationSystem:
    def __init__(self,
                 corrupt_for_test=False,
                 intercept_time=0.3,
                 sigma_a=1500.0,
                 lost_uncertainty_cm=3000.0,
                 delay_compensation=3.0):

        self.drone_ekf = EKF(sigma_a=sigma_a)
        self.target_ekf = EKF(sigma_a=sigma_a)

        self.prefilter = GNSSPrefilter()
        self.corruptor = GNSSCorruptor(enabled=corrupt_for_test)

        self.wp = WaypointController(intercept_time=intercept_time)

        self.delay_compensation = delay_compensation

        self.lost_uncertainty_cm = lost_uncertainty_cm
        self._last_t = None
        self._last_t_meas = None  
        self._target_miss = 0     
        self._stale_streak = 0  
        self._last_valid_speed = 0.0  

    def process(self, telemetry):
        now = time.monotonic()
        if self._last_t is None:
            dt = self.drone_ekf.dt_nominal
        else:
            dt = now - self._last_t
        self._last_t = now

        d_pos_raw = tuple(telemetry["drone"]["position"])
        t_pos_raw = tuple(telemetry["target"]["position"])

        d_meas = self.prefilter.prefilter(d_pos_raw)
        self.drone_ekf.step(d_meas, dt=dt)

        t_corrupted = self.corruptor.corrupt(t_pos_raw)
        t_meas = self.prefilter.prefilter(t_corrupted)

        is_stale = (self.target_ekf.initialized
                    and t_meas is not None and self._last_t_meas is not None
                    and t_meas == self._last_t_meas)
        if is_stale:

            self.target_ekf.step(None, dt=dt)
            accepted = False
            self._stale_streak += 1
        else:
            self._stale_streak = 0
            _, accepted = self.target_ekf.step(t_meas, dt=dt, t_now=now)
            if t_meas is None or not accepted:
                self._target_miss += 1     # gercek kayip: dropout veya outlier
            else:
                self._target_miss = 0
                self._last_t_meas = t_meas

        out = self.wp.compute(self.drone_ekf, self.target_ekf,
                              extra_lead=self.delay_compensation)

        spd = out["target_speed"]
        if spd > 50.0:
            self._last_valid_speed = spd
        elif self._last_valid_speed > 0:
            out["target_speed"] = self._last_valid_speed   # gecici 0'i maskele

        unc = self.target_ekf.position_uncertainty()
        out["gps_ok"] = (unc < self.lost_uncertainty_cm
                         and self._target_miss < 30
                         and self._stale_streak < 40)
        out["target_uncertainty"] = unc
        out["dt"] = dt
        return out