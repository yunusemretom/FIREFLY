"""
PID controller for drone pitch/speed.
"""

# pid_pitch.py
import math
from .pid_core import PID, SlewLimiter, clamp


class PitchController:
    def __init__(self, config):
        op = config["outer_pid"]["speed"]
        self.speed_pid = PID(
            kp=op["kp"],
            ki=op["ki"],
            kd=op["kd"],
            clamp=(op["clamp_min"], op["clamp_max"]),
            integral_limit=op["int_limit"],
        )

        ip = config["inner_pid"]["pitch"]
        self.inner_pid = PID(kp=ip["kp"], ki=ip["ki"], kd=ip["kd"], clamp=(-1.0, 1.0))

        self.slew = SlewLimiter(config["slew"]["pitch_max_change_per_sec"])
        self.min_pitch = config["limits"]["min_pitch"]
        self.max_pitch = config["limits"]["max_pitch"]
        # Stick tam skalası (deg): canlı kalibrasyonla ölçüldü — sim SIMETRIK,
        # stick*60 = gerçek eğim (0.25→15°, 0.50→30°, 0.75→45°).
        self.full_scale = float(config["limits"].get("pitch_full_scale_deg", 60.0))
        # Fren (burun-yukarı) açı tavanı: kamera hedefi görüşte tutsun diye
        # max_pitch'ten ayrı, daha sıkı bir tavan. Sert fren yerine hafif
        # burun-yukarı + sürükleme ile yavaşla; mesafe hız kanalıyla tutulur.
        self.brake_pitch_max = float(config["limits"].get("max_brake_pitch", 12.0))

    def calculate(
        self, state, target_speed_cmd, drone_speed, target_elevation, drone_pitch, yaw_error, dt
    ):
        if state == "KALKIS":
            raw_desired = 0.0
        else:
            speed_error_ms = (target_speed_cmd - drone_speed) / 100.0
            speed_adj = self.speed_pid.calculate(speed_error_ms, dt)

            raw_desired = target_elevation - speed_adj
            # Fren tarafı (burun yukarı, pozitif) kamera-görüş tavanıyla sınırlı
            raw_desired = clamp(raw_desired, self.min_pitch,
                                min(self.max_pitch, self.brake_pitch_max))

        smooth_desired = self.slew.update(raw_desired, dt)

        # Normalize to [-1, 1] angle command — SİMETRİK tam skala.
        # ESKİ HATA: pozitif açı max_pitch'e (20) bölünüyordu; sim stick'i
        # ±full_scale (60°) yorumladığından fren isteği 3 KAT aşırı eğim
        # üretiyordu (14° istek → 0.71 stick → 43° gerçek geriye yatış →
        # hedef kamera görüşünden çıkıyordu).
        pitch_cmd = smooth_desired / self.full_scale if self.full_scale else 0.0

        # Scale pitch by yaw alignment: prevents crabbing sideways before facing the target.
        # cos(0°)=1 (aligned → full pitch), cos(90°)=0 (perpendicular → reduced pitch).
        # Taban 0.3: minimum %30 pitch yetkisi her zaman korunur — yaw hatası büyükken
        # bile drone ileri hızını tamamen kaybetmez. Bu, "hızlanamıyor → geride kalıyor
        # → yaw hatası büyüyor → daha da hızlanamıyor" kısır döngüsünü kırar.
        yaw_scale = max(0.3, math.cos(math.radians(min(abs(yaw_error), 80.0))))
        pitch_cmd = clamp(pitch_cmd * yaw_scale, -1.0, 1.0)

        return smooth_desired, pitch_cmd
