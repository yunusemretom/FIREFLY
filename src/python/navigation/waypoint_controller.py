
import math


class WaypointController:
    def __init__(self, intercept_time=0.3, camera_tilt=0.0,
                 adaptive_lead=True, lead_min=0.1, lead_max=0.8,
                 use_intercept=True, max_lead=1.2, smooth_a=0.15):

        self.intercept_time = intercept_time
        self.camera_tilt = camera_tilt
        self.adaptive_lead = adaptive_lead
        self.lead_min = lead_min
        self.lead_max = lead_max
        self.use_intercept = use_intercept
        self.max_lead = max_lead
        self.smooth_a = smooth_a
        self._t_use_smooth = None       # intercept zamani yumusatma durumu

    def _lead_time(self, dist_3d, closing_speed):

        if not self.adaptive_lead:
            return self.intercept_time
        if closing_speed <= 1.0:
            return self.intercept_time
        t = dist_3d / closing_speed          # kabaca temas suresi (s)
        t = max(self.lead_min, min(self.lead_max, t))
        # yakin mesafe sonumlemesi: 4000cm altinda lead'i lineer kis
        if dist_3d < 4000.0:
            t *= max(0.0, dist_3d / 4000.0)
        return t

    def _intercept_time(self, rx, ry, rz, t_vel, drone_speed):

        if drone_speed < 1.0:
            return None
        tvx, tvy, tvz = t_vel
        a = (tvx*tvx + tvy*tvy + tvz*tvz) - drone_speed*drone_speed
        b = 2.0 * (rx*tvx + ry*tvy + rz*tvz)
        c = rx*rx + ry*ry + rz*rz

        if abs(a) < 1e-6:                 # drone hizi ~ hedef hizi: lineer cozum
            if abs(b) < 1e-6:
                return None
            t = -c / b
            return t if t > 0 else None

        disc = b*b - 4*a*c
        if disc < 0:
            return None                    # gercek kesisim yok
        sq = math.sqrt(disc)
        t1 = (-b + sq) / (2*a)
        t2 = (-b - sq) / (2*a)
        cand = [t for t in (t1, t2) if t > 0]
        if not cand:
            return None
        t = min(cand)
        return min(t, 8.0)

    def compute(self, drone_ekf, target_ekf, extra_lead=0.0):
        d_pos = drone_ekf.get_position()
        d_vel = drone_ekf.get_velocity()

        t_pos = target_ekf.get_position()
        t_vel = target_ekf.get_velocity()

        rx = t_pos[0] - d_pos[0]
        ry = t_pos[1] - d_pos[1]
        rz = t_pos[2] - d_pos[2]
        r_now = math.sqrt(rx * rx + ry * ry + rz * rz)

        rvx = t_vel[0] - d_vel[0]
        rvy = t_vel[1] - d_vel[1]
        rvz = t_vel[2] - d_vel[2]
        if r_now > 1e-6:
            closing = -(rx * rvx + ry * rvy + rz * rvz) / r_now
        else:
            closing = 0.0

        drone_speed = drone_ekf.get_speed()
        t_intercept = self._intercept_time(rx, ry, rz, t_vel, drone_speed)

        if t_intercept is not None:
            t_use = t_intercept
        else:
            t_use = self._lead_time(r_now, closing)

        MAX_LEAD = 1.2
        t_use = max(0.0, min(t_use, MAX_LEAD))

        if r_now < 12000.0:
            t_use *= max(0.0, (r_now - 2000.0) / 10000.0)

        if self._t_use_smooth is None:
            self._t_use_smooth = t_use
        else:
            a = 0.08   # cok dusuk = cok yumusak (yaw/pitch titremesini azaltir)
            self._t_use_smooth = (1 - a) * self._t_use_smooth + a * t_use
        t_use = self._t_use_smooth
        t_lead = t_use

        total_ahead = min(t_use + extra_lead, 3.5)
        tgt_future = target_ekf.predict_future(total_ahead)

        t_pos_comp = target_ekf.predict_future(extra_lead)

        dx = tgt_future[0] - d_pos[0]
        dy = tgt_future[1] - d_pos[1]
        dz = tgt_future[2] - d_pos[2]
        dist_2d = math.sqrt(dx * dx + dy * dy)
        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

        target_yaw = math.degrees(math.atan2(dy, dx))
        denom = dist_2d if dist_2d > 1e-6 else 1e-6
        target_pitch_world = math.degrees(math.atan2(dz, denom))

        return {
            "waypoint": tgt_future,        # filtrelenmis + ongorulu hedef konum
            "drone_pos": d_pos,
            "target_pos": t_pos_comp,      # gecikme-telafili anlik hedef (lead'siz)
            "dx": dx, "dy": dy, "dz": dz,
            "dist_2d": dist_2d,
            "dist_3d": dist_3d,
            "r_now": r_now,                # ongorusuz anlik mesafe
            "closing_speed": closing,      # cm/s, >0 yaklasiyor
            "lead_time": t_lead,
            "target_yaw": target_yaw,            # derece
            "target_pitch_world": target_pitch_world,
            "camera_tilt": self.camera_tilt,
            "target_speed": target_ekf.get_speed(),
            "drone_speed": drone_ekf.get_speed(),
        }