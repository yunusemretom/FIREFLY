"""
Pursuit guidance controller — "5 metre arkadan takip"

Hedefin hız vektörünün tam tersine follow_distance_cm uzaklıkta sanal bir
"takip noktası" hesaplar ve dronu o noktada tutar.

Hız düzenlemesi (geçme engeli — PID):
  s = imzalı mesafe = (drone − takip_noktası) · hedef_yönü
  s > 0 → drone çok yakın / geçmek üzere → yavaşla
  s = 0 → doğru konumda              → hedef hızını tut
  s < 0 → drone geri kaldı           → hızlan

  dist_pid hatası = −s   (sıfır = mükemmel konum)
  target_speed = hedef_hızı + dist_pid.calculate(−s)   (min/max ile sınırlı)

Spin önleme:
  YAW her zaman HEDEFe bakar — takip noktasına değil.
  Takip noktası drone'un arkasına düştüğünde 180° dönme ihtiyacı olmaz;
  drone sadece ileri hızını kısarak hedefin kendiliğinden uzaklaşmasını bekler.
"""

import math

from .pid_core import PID, SlewLimiter, clamp, wrap_180


class PursuitController:
    """
    Dronu hedefin tam `follow_distance_cm` arkasında tutar.

    Kullanım
    --------
    ctrl = PursuitController(config)

    # her döngüde:
    cmds, telem = ctrl.calculate(
        drone_pos, drone_rot, drone_speed,
        target_pos, target_vel,   # target_vel → EKF'den gelir
        dt
    )
    thr, pitch, roll, yaw = cmds
    drone.set_control_surfaces(thr, -pitch, -roll, yaw, True)
    """

    def __init__(self, config: dict):
        pur = config.get("pursuit", {})
        self.follow_dist: float = float(pur.get("follow_distance_cm", 500.0))
        self.min_speed:   float = float(pur.get("min_speed_cms", 200.0))
        self.max_speed:   float = float(pur.get("max_speed_cms", 2800.0))

        # Dış döngü: imzalı mesafe hatası → hız düzeltmesi
        dp = pur.get("distance_pid", {})
        self._dist_pid = PID(
            kp=float(dp.get("kp", 0.5)),
            ki=float(dp.get("ki", 0.02)),
            kd=float(dp.get("kd", 0.15)),
            integral_limit=float(dp.get("int_limit", 200.0)),
        )

        # Yaw
        yp = config["inner_pid"]["yaw"]
        self._yaw_pid = PID(kp=yp["kp"], ki=yp["ki"], kd=yp["kd"], clamp=(-1.0, 1.0))

        # Pitch — iç hız döngüsü
        sp = config["outer_pid"]["speed"]
        self._speed_pid = PID(
            kp=sp["kp"], ki=sp["ki"], kd=sp["kd"],
            clamp=(sp["clamp_min"], sp["clamp_max"]),
            integral_limit=sp["int_limit"],
        )
        self._slew = SlewLimiter(config["slew"]["pitch_max_change_per_sec"])

        # Throttle — irtifa
        ap = config["outer_pid"]["altitude"]
        self._alt_pid = PID(
            kp=ap["kp"], ki=ap["ki"], kd=ap["kd"],
            clamp=(ap["clamp_min"], ap["clamp_max"]),
            integral_limit=ap["int_limit"],
        )
        self._base_throttle: float = float(config["flight"]["base_throttle"])
        self._min_t: float = float(config["limits"]["min_throttle"])
        self._max_t: float = float(config["limits"]["max_throttle"])

        self._camera_tilt: float = float(config["flight"]["camera_tilt"])
        self._min_pitch:   float = float(config["limits"]["min_pitch"])
        self._max_pitch:   float = float(config["limits"]["max_pitch"])

    # ------------------------------------------------------------------

    def calculate(
        self,
        drone_pos:   tuple,
        drone_rot:   tuple,
        drone_speed: float,
        target_pos:  tuple,
        target_vel:  tuple,
        dt:          float,
    ):
        """
        Args
        ----
        drone_pos, drone_rot : (x,y,z) cm ve (roll,pitch,yaw) derece
        drone_speed          : cm/s skaler
        target_pos           : (x,y,z) cm  — EKF tahmini veya truth
        target_vel           : (vx,vy,vz) cm/s — EKF'den (≈1500 cm/s)
        dt                   : döngü süresi (s)

        Returns
        -------
        commands : (throttle, pitch, roll, yaw)  her biri [-1, 1]
        telemetry: dict  — loglama için
        """
        # 1. Hedef yönü (birim vektör)
        vx, vy, vz = target_vel
        speed_mag = math.sqrt(vx**2 + vy**2 + vz**2)
        if speed_mag > 1.0:
            ux, uy, uz = vx / speed_mag, vy / speed_mag, vz / speed_mag
        else:
            ux = uy = uz = 0.0  # EKF ısınmadı → offset yok, direkt hedefe

        # 2. Takip noktası (sadece mesafe hesabı için kullanılır)
        fx = target_pos[0] - ux * self.follow_dist
        fy = target_pos[1] - uy * self.follow_dist
        fz = target_pos[2] - uz * self.follow_dist

        # 3. İmzalı mesafe s
        #    s > 0 → drone, takip noktasının "ilerisinde" (çok yakın / geçti)
        #    s < 0 → drone, takip noktasının "gerisinde" (geri kaldı)
        rx = drone_pos[0] - fx
        ry = drone_pos[1] - fy
        rz = drone_pos[2] - fz
        signed_dist = rx * ux + ry * uy + rz * uz

        # 4. Dış PID: mesafe hatası → hız hedefi
        #    Hata = 0 − s  →  s < 0 ise pozitif hata → hızlan
        #                      s > 0 ise negatif hata → yavaşla
        dist_correction = self._dist_pid.calculate(-signed_dist, dt)
        target_speed_cmd = speed_mag + dist_correction
        target_speed_cmd = clamp(target_speed_cmd, self.min_speed, self.max_speed)

        # 5. YAW: her zaman HEDEFE bak (takip noktasına değil!)
        #    Takip noktası drone'un arkasına düştüğünde 180° dönme olmaz.
        #    Drone ileri hızını kısarak hedefin uzaklaşmasını bekler.
        tdx = target_pos[0] - drone_pos[0]
        tdy = target_pos[1] - drone_pos[1]
        tdz = target_pos[2] - drone_pos[2]
        bearing   = math.degrees(math.atan2(tdy, tdx))
        yaw_error = wrap_180(bearing - drone_rot[2])
        yaw_cmd   = clamp(self._yaw_pid.calculate(yaw_error, dt), -1.0, 1.0)

        # 6. Throttle: hedefin irtifasına eşleş
        dz_m     = clamp(tdz / 100.0, -5.0, 5.0)
        alt_corr = self._alt_pid.calculate(dz_m, dt)
        throttle = clamp(self._base_throttle + alt_corr, self._min_t, self._max_t)

        # 7. Pitch: iç hız döngüsü + hedef elevasyon açısı
        dist_2d       = math.sqrt(tdx**2 + tdy**2)
        target_elev   = math.degrees(math.atan2(tdz, max(dist_2d, 1.0)))
        speed_err_ms  = (target_speed_cmd - drone_speed) / 100.0
        speed_adj     = self._speed_pid.calculate(speed_err_ms, dt)

        camera_center = target_elev - self._camera_tilt
        raw_pitch     = clamp(camera_center - speed_adj, self._min_pitch, self._max_pitch)
        smooth_pitch  = self._slew.update(raw_pitch, dt)

        if smooth_pitch < 0:
            pitch_cmd = smooth_pitch / abs(self._min_pitch) if self._min_pitch else 0.0
        else:
            pitch_cmd = smooth_pitch / abs(self._max_pitch) if self._max_pitch else 0.0

        # Dönüş sırasında pitch ölçekleme: 90°'de %30 minimum → tam durmaz
        yaw_scale = max(0.3, math.cos(math.radians(yaw_error)))
        pitch_cmd = clamp(pitch_cmd * yaw_scale, -1.0, 1.0)

        commands  = (throttle, pitch_cmd, 0.0, yaw_cmd)
        telemetry = {
            "follow_point":   (fx, fy, fz),
            "signed_dist_cm": signed_dist,
            "target_speed":   target_speed_cmd,
            "smooth_pitch":   smooth_pitch,
            "yaw_error":      yaw_error,
        }
        return commands, telemetry
