"""
Yaw PID Controller — Gyro Damping destekli.

Opsiyonel `yaw_rate` (ölçülen yaw hızı, deg/s) parametresi geçirildiğinde,
gürültülü error-derivative yerine drone'un gerçek dönüş hızından sönümleme
yapılır (pursuit_guidance.py v2 ile aynı prensip). Bu, 50 Hz kontrol
döngüsünde GPS gürültüsünden kaynaklanan yaw salınımını ortadan kaldırır.

ÖNEMLİ: Ham yaw rate 50 Hz'de çok gürültülü olduğu için mutlaka düşük geçiren
filtre (tau=0.1s) uygulanır. Pursuit v2'nin kanıtladığı gibi, filtresiz ham
derivative, damping terimini patlatıp bang-bang salınımı üretir.

yaw_rate verilmediğinde eski PID davranışı korunur (geriye uyumluluk).
"""

import math
from .pid_core import PID, wrap_180, clamp


class YawController:
    def __init__(self, config):
        ip = config["inner_pid"]["yaw"]
        self.inner_pid = PID(kp=ip["kp"], ki=ip["ki"], kd=ip["kd"], clamp=(-1.0, 1.0))
        # Gyro damping parametreleri
        self.kp = ip["kp"]
        self.ki = ip.get("ki", 0.0)
        # kd_gyro: gyro damping katsayısı — error-derivative kd'den AYRI.
        # P:D oranı pursuit v2 ile uyumlu: kp/kd_gyro ≈ 8.75
        # (pursuit v2: kp=0.07, kd=0.008 → 8.75 ; burada: 0.035/0.004 = 8.75)
        self.kd_gyro = float(ip.get("kd_gyro", 0.004))
        # Integral SINIRI (deg·s): gyro-damping yolundaki _integral daha önce
        # sınırsızdı — bekleme/hover fazında dakikalarca hata biriktirip kilit
        # sonrası yaw'ı satüre ediyordu (windup).
        self.int_limit = float(ip.get("int_limit", 150.0))
        self._integral = 0.0
        # Düşük geçiren filtre (tau=0.1s) — ham 50 Hz yaw rate'i yumuşatır
        self._rate_filt = 0.0
        self._rate_filter_tau = 0.1

    def reset(self):
        """Dönüş komutu verilmeyen fazlarda (bekleme/near-field) çağrılır:
        HER İKİ yolun integral/türev durumunu sıfırlar. Runner'ın yalnızca
        inner_pid.integral'i sıfırlaması gyro yolundaki windup'ı temizlemiyordu."""
        self._integral = 0.0
        self.inner_pid.integral = 0.0
        self.inner_pid.first_run = True

    def calculate(self, dx, dy, drone_yaw, dt, yaw_rate=None):
        """Yaw komutu hesaplar.

        Args:
            dx, dy: guide noktasına göre ofset (guide_x - drone_x, guide_y - drone_y)
            drone_yaw: drone heading (derece)
            dt: zaman adımı (saniye)
            yaw_rate: ölçülen yaw hızı (deg/s), verilirse gyro damping kullanılır
        Returns:
            (yaw_error, yaw_cmd) — derece cinsinden hata ve [-1, 1] stick komutu
        """
        target_bearing = math.degrees(math.atan2(dy, dx))
        yaw_error = wrap_180(target_bearing - drone_yaw)

        if yaw_rate is not None:
            # GYRO DAMPING: P + I − kd_gyro · filtered_yaw_rate
            # 1) Integral birikimi (daima sınırlı — windup koruması)
            self._integral += yaw_error * dt
            self._integral = clamp(self._integral, -self.int_limit, self.int_limit)
            # 2) Yaw rate'e düşük geçiren filtre uygula (tau=0.1s)
            #    Ham 50 Hz derivative çok gürültülü; filtre olmadan D terimi
            #    P terimini bastırıp bang-bang üretir (v1 hatası).
            alpha = 1.0 - math.exp(-dt / self._rate_filter_tau)
            self._rate_filt += alpha * (yaw_rate - self._rate_filt)
            # 3) P + I − D(gyro)
            output = (self.kp * yaw_error
                      + self.ki * self._integral
                      - self.kd_gyro * self._rate_filt)
            yaw_cmd = clamp(output, -1.0, 1.0)
        else:
            # Klasik error-derivative PID (geriye uyumluluk)
            yaw_cmd = clamp(self.inner_pid.calculate(yaw_error, dt), -1.0, 1.0)

        return yaw_error, yaw_cmd
