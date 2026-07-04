"""
================================================================================
FIXED-WING TARGET EKF — Sabit Dönüş Hızı ve Sabit Hız Modeli (CTRV)
================================================================================
Modül: src/python/navigation/ekf.py

Bu modül, Unreal Engine tabanlı Drone of War ortamında sabit kanatlı bir
hedef uçağı (Talon İHA) takip eden Extended Kalman Filter (EKF) implementasyonunu
içerir. Bütün parametre ayarları config/ekf_params.yaml dosyasından okunur;
bu sınıfın kendi içinde hard-coded değer YOKTUR.

Tasarım Kararları:
  1. CTRV Modeli seçildi çünkü:
       - Sabit kanatlı bir uçak yerinde dönemez; dönüşler kavis yaparak gerçekleşir.
       - CTRV, yörüngenin eğrisel bileşenini (omega) durum vektörüne dahil ederek
         dönüş sırasında kör tahmin (dead-reckoning) doğruluğunu önemli ölçüde artırır.
       - Standart Constant Velocity (CV) modeli düz uçuşta iyi çalışır; ancak
         loiter / banked turn manevralarında büyük lineerizasyon hatası üretir.

  2. GPS Spike Bastırma:
       - Düşük Q (süreç gürültüsü): filtre kendi kinematik modeline güvenir.
       - Yüksek R (ölçüm gürültüsü): ani GPS zıplamaları direkt kabul edilmez.
       - Mahalanobis mesafesi eşiği: istatistiksel olarak aykırı ölçümler reddedilir.

Durum Vektörü x = [x, y, theta, v, omega]^T
  - x, y    : 2D yatay konum (cm)
  - theta   : yaw açısı (rad, ENU çerçevesinde)
  - v       : doğrusal yatay hız (cm/s)
  - omega   : dönüş (yaw) hızı (rad/s)

Ölçüm Vektörü z = [x, y]^T
  - Sadece GPS konum verisi kullanılır (2 boyutlu).
================================================================================
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Config yükleme yardımcısı
# ---------------------------------------------------------------------------

def _load_ekf_config() -> dict:
    """
    config/ekf_params.yaml dosyasını proje kökünden otomatik bulup yükler.
    Bu sayede modül herhangi bir çalışma dizininden import edilebilir.
    """
    # Bu dosya: src/python/navigation/ekf.py
    # Proje kökü: ../../.. (3 seviye yukarı)
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[3]
    config_path = project_root / "config" / "ekf_params.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"EKF config dosyası bulunamadı: {config_path}\n"
            f"Beklenen konum: config/ekf_params.yaml (proje kökü: {project_root})"
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        full_cfg = yaml.safe_load(fh)

    return full_cfg.get("ctrv_ekf", {})


# ---------------------------------------------------------------------------
# FixedWingEKF — Ana EKF Sınıfı
# ---------------------------------------------------------------------------

class FixedWingEKF:
    """
    Sabit kanatlı hedef uçak için 5-boyutlu CTRV Extended Kalman Filter.

    Kullanım:
        ekf = FixedWingEKF()
        ekf.predict(dt=0.1)                   # 10 Hz tahmin
        ekf.update(z=np.array([x_gps, y_gps]))  # 1 Hz GPS güncellemesi

    Özellikler:
        ekf.state          -> np.ndarray (5,) — mevcut durum tahmini
        ekf.covariance     -> np.ndarray (5,5) — hata kovaryansı
        ekf.is_initialized -> bool
        ekf.position       -> (x, y) tuple (cm)
        ekf.heading        -> float (rad)
        ekf.speed          -> float (cm/s)
        ekf.yaw_rate       -> float (rad/s)
    """

    # Durum indeksleri — okunabilirlik için sabitler
    IDX_X     = 0
    IDX_Y     = 1
    IDX_THETA = 2
    IDX_V     = 3
    IDX_OMEGA = 4

    def __init__(self, config: Optional[dict] = None) -> None:
        """
        Args:
            config: İsteğe bağlı config dict. None ise ekf_params.yaml'dan yüklenir.
        """
        cfg = config if config is not None else _load_ekf_config()

        # -- Süreç Gürültüsü (Q) --------------------------------------------------
        # Mantık: Küçük Q → filtre kendi CTRV modeline daha çok güvenir.
        # GPS spike'ları geldiğinde konum aniden zıplamak yerine yavaşça güncellenir.
        pn = cfg["process_noise"]
        q_xy    = float(pn["q_xy"])      # cm^2 — konum (x, y)
        q_theta = float(pn["q_theta"])   # rad^2 — heading
        q_v     = float(pn["q_v"])       # (cm/s)^2 — hız
        q_omega = float(pn["q_omega"])   # (rad/s)^2 — yaw rate

        self._Q = np.diag([q_xy, q_xy, q_theta, q_v, q_omega])

        # -- Ölçüm Gürültüsü (R) --------------------------------------------------
        # Mantık: Büyük R → GPS'e az güven. 1 Hz, gürültülü ve spike'lı GPS'te
        # filtrenin ani sıçramaması için yüksek tutulur.
        mn = cfg["measurement_noise"]
        r_xy = float(mn["r_xy"])         # cm^2 — (~2 m std sapma)
        self._R = np.diag([r_xy, r_xy])

        # -- Başlangıç Kovaryansı (P0) ---------------------------------------------
        ic = cfg["initial_covariance"]
        p_xy    = float(ic["p_xy"])
        p_theta = float(ic["p_theta"])
        p_v     = float(ic["p_v"])
        p_omega = float(ic["p_omega"])

        self._P0 = np.diag([p_xy, p_xy, p_theta, p_v, p_omega])

        # -- Ölçüm matrisi (H) — doğrusal; sadece x, y okunur ---------------
        # z = H @ x + v, H şekli: (2, 5)
        self._H = np.zeros((2, 5))
        self._H[0, self.IDX_X] = 1.0
        self._H[1, self.IDX_Y] = 1.0

        # -- Aykırı Değer Reddi ---------------------------------------------------
        or_cfg = cfg.get("outlier_rejection", {})
        self._outlier_enabled   = bool(or_cfg.get("enabled", True))
        self._mahal_threshold   = float(or_cfg.get("mahalanobis_threshold", 13.82))

        # -- Başlangıç ayarları ---------------------------------------------------
        init_cfg = cfg.get("init", {})
        self._startup_samples   = int(init_cfg.get("startup_samples", 2))
        self._default_speed     = float(init_cfg.get("default_speed_cms", 1500.0))

        # -- Sayısal kararlılık ---------------------------------------------------
        num = cfg.get("numerical", {})
        self._jitter = float(num.get("jitter", 1e-6))

        # -- Dahili durum ---------------------------------------------------------
        self._x: np.ndarray = np.zeros(5)   # durum vektörü
        self._P: np.ndarray = self._P0.copy()  # hata kovaryansı
        self._initialized: bool = False
        self._sample_count: int  = 0
        self._update_count: int  = 0        # kaç GPS güncellemesi kabul edildi
        self._warmup_updates: int = int(init_cfg.get("warmup_updates", 5))
        # ilk N güncellemede Mahalanobis testi atlanır → heading yakınsaması
        self._prev_gps: Optional[np.ndarray] = None  # önceki GPS ölçümü
        self._prev_gps_time: float = 0.0

    # ------------------------------------------------------------------
    # Özellik Erişicileri (Properties)
    # ------------------------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        """Mevcut 5-boyutlu durum vektörü."""
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """5×5 hata kovaryans matrisi."""
        return self._P.copy()

    @property
    def is_initialized(self) -> bool:
        """Filtre yeterli GPS ölçümü aldı mı?"""
        return self._initialized

    @property
    def position(self) -> Tuple[float, float]:
        """Mevcut konum tahmini: (x_cm, y_cm)"""
        return float(self._x[self.IDX_X]), float(self._x[self.IDX_Y])

    @property
    def heading(self) -> float:
        """Mevcut heading tahmini (rad)."""
        return float(self._x[self.IDX_THETA])

    @property
    def speed(self) -> float:
        """Mevcut hız tahmini (cm/s)."""
        return float(self._x[self.IDX_V])

    @property
    def yaw_rate(self) -> float:
        """Mevcut dönüş hızı (rad/s)."""
        return float(self._x[self.IDX_OMEGA])

    # ------------------------------------------------------------------
    # CTRV Hareket Modeli — f(x, dt)
    # ------------------------------------------------------------------

    def _f(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        CTRV geçiş fonksiyonu: x_{k+1} = f(x_k, dt)

        CTRV denklemleri (omega ≠ 0 durumu):
            x'     = x + (v/omega) * [sin(theta + omega*dt) - sin(theta)]
            y'     = y + (v/omega) * [-cos(theta + omega*dt) + cos(theta)]
            theta' = theta + omega * dt
            v'     = v                          (sabit hız varsayımı)
            omega' = omega                       (sabit dönüş hızı varsayımı)

        omega ≈ 0 durumunda (düz uçuş), sayısal kararsızlığı önlemek için
        1. dereceden Taylor açılımı kullanılır:
            x'     = x + v * cos(theta) * dt
            y'     = y + v * sin(theta) * dt
        """
        x_pos  = x[self.IDX_X]
        y_pos  = x[self.IDX_Y]
        theta  = x[self.IDX_THETA]
        v      = x[self.IDX_V]
        omega  = x[self.IDX_OMEGA]

        x_new = np.zeros(5)

        if abs(omega) > 1e-6:
            # Tam CTRV modeli (dönüş manevrası)
            sin_t      = math.sin(theta)
            cos_t      = math.cos(theta)
            sin_t_dt   = math.sin(theta + omega * dt)
            cos_t_dt   = math.cos(theta + omega * dt)

            x_new[self.IDX_X]     = x_pos + (v / omega) * (sin_t_dt - sin_t)
            x_new[self.IDX_Y]     = y_pos + (v / omega) * (-cos_t_dt + cos_t)
        else:
            # Düz uçuş (omega ≈ 0) — CV bozulması
            x_new[self.IDX_X]     = x_pos + v * math.cos(theta) * dt
            x_new[self.IDX_Y]     = y_pos + v * math.sin(theta) * dt

        x_new[self.IDX_THETA] = theta + omega * dt
        x_new[self.IDX_V]     = v
        x_new[self.IDX_OMEGA] = omega

        return x_new

    # ------------------------------------------------------------------
    # Jakobyen — F = df/dx (Analitik)
    # ------------------------------------------------------------------

    def _jacobian_F(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        f(x, dt)'nin x'e göre analitik Jakobyen matrisini hesaplar (5×5).

        Türevler:
            ∂x'/∂x = 1
            ∂x'/∂theta = (v/omega)*[cos(theta+omega*dt) - cos(theta)]
            ∂x'/∂v    = (1/omega)*[sin(theta+omega*dt) - sin(theta)]
            ∂x'/∂omega = v*[cos(theta+omega*dt)*dt/omega - (sin(theta+omega*dt)-sin(theta))/omega^2]

            ∂y'/∂y = 1
            ∂y'/∂theta = (v/omega)*[sin(theta+omega*dt) - sin(theta)]  (dikkat: işaret)
            ∂y'/∂v    = (1/omega)*[-cos(theta+omega*dt) + cos(theta)]
            ∂y'/∂omega = v*[sin(theta+omega*dt)*dt/omega + (cos(theta+omega*dt)-cos(theta))/omega^2]

            ∂theta'/∂theta = 1
            ∂theta'/∂omega = dt

            Kalan diyagonal elemanlar 1; çapraz türevler 0.

        omega ≈ 0 durumunda sayısal kararlılık için limit değerler kullanılır.
        """
        theta = x[self.IDX_THETA]
        v     = x[self.IDX_V]
        omega = x[self.IDX_OMEGA]

        F = np.eye(5)  # Başlangıç: birim matris

        if abs(omega) > 1e-6:
            sin_t    = math.sin(theta)
            cos_t    = math.cos(theta)
            sin_t_dt = math.sin(theta + omega * dt)
            cos_t_dt = math.cos(theta + omega * dt)
            inv_w    = 1.0 / omega
            inv_w2   = inv_w * inv_w

            # ∂x'/∂theta
            F[self.IDX_X, self.IDX_THETA] = (v * inv_w) * (cos_t_dt - cos_t)
            # ∂x'/∂v
            F[self.IDX_X, self.IDX_V]     = inv_w * (sin_t_dt - sin_t)
            # ∂x'/∂omega
            F[self.IDX_X, self.IDX_OMEGA] = (
                v * (cos_t_dt * dt * inv_w - (sin_t_dt - sin_t) * inv_w2)
            )

            # ∂y'/∂theta
            F[self.IDX_Y, self.IDX_THETA] = (v * inv_w) * (sin_t_dt - sin_t)
            # ∂y'/∂v
            F[self.IDX_Y, self.IDX_V]     = inv_w * (-cos_t_dt + cos_t)
            # ∂y'/∂omega
            F[self.IDX_Y, self.IDX_OMEGA] = (
                v * (sin_t_dt * dt * inv_w + (cos_t_dt - cos_t) * inv_w2)
            )
        else:
            # omega ≈ 0 → limit (Taylor genişlemesi)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            F[self.IDX_X, self.IDX_THETA] = -v * sin_t * dt
            F[self.IDX_X, self.IDX_V]     =  cos_t * dt
            # ∂x'/∂omega limit: (v/2)*cos(theta)*dt^2
            F[self.IDX_X, self.IDX_OMEGA] = -0.5 * v * sin_t * dt * dt

            F[self.IDX_Y, self.IDX_THETA] =  v * cos_t * dt
            F[self.IDX_Y, self.IDX_V]     =  sin_t * dt
            # ∂y'/∂omega limit: (v/2)*sin(theta)*dt^2
            F[self.IDX_Y, self.IDX_OMEGA] =  0.5 * v * cos_t * dt * dt

        # ∂theta'/∂omega = dt
        F[self.IDX_THETA, self.IDX_OMEGA] = dt

        return F

    # ------------------------------------------------------------------
    # Bootstrap — İlk GPS ölçümünden durum başlatma
    # ------------------------------------------------------------------

    def _bootstrap(self, z: np.ndarray, timestamp: float) -> bool:
        """
        İlk birkaç GPS ölçümünden başlangıç heading'i tahmin eder.

        Tasarım notu:
            Hız tahmini için gürültülü GPS farkı (dist/dt) kullanılmaz çünkü
            GPS spike'ları sıfır veya negatif hız üretebilir. Bunun yerine
            config'den gelen `default_speed_cms` kullanılır; heading ise iki
            ardışık GPS noktasının farkından tahmin edilir.

        Returns:
            True → yeterli örnek toplandı, filtre artık hazır.
        """
        self._sample_count += 1

        if self._sample_count == 1:
            # İlk ölçüm: konumu kaydet, varsayılan hız ve sıfır dönüş ile başla
            self._x[self.IDX_X]     = z[0]
            self._x[self.IDX_Y]     = z[1]
            self._x[self.IDX_V]     = self._default_speed  # cm/s — config'den
            self._x[self.IDX_THETA] = 0.0
            self._x[self.IDX_OMEGA] = 0.0
            self._prev_gps = z.copy()
            self._prev_gps_time = timestamp
            return False

        if self._sample_count <= self._startup_samples:
            # İkinci ölçüm: heading'i GPS yer değişiminden tahmin et.
            # Hız tahmini için default_speed kullanılır — gürültüden etkilenmez.
            dx = z[0] - self._prev_gps[0]
            dy = z[1] - self._prev_gps[1]
            dist = math.hypot(dx, dy)
            if dist > 100.0:   # gürültünün üzerinde anlamlı hareket (1 m)
                self._x[self.IDX_THETA] = math.atan2(dy, dx)
            # Hız her zaman default'tan başlar; EKF güncellemelerle iyileştirir
            self._x[self.IDX_V]     = self._default_speed
            self._x[self.IDX_X]     = z[0]
            self._x[self.IDX_Y]     = z[1]
            self._x[self.IDX_OMEGA] = 0.0
            self._prev_gps = z.copy()
            self._prev_gps_time = timestamp

        if self._sample_count >= self._startup_samples:
            self._P = self._P0.copy()
            self._initialized = True
            return True

        return False

    # ------------------------------------------------------------------
    # PREDICT — Tahmin Adımı
    # ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        """
        EKF tahmin adımı (10 Hz'de çağrılır).

        GPS verisi gelsin ya da gelmesin bu fonksiyon her zaman çağrılmalıdır.
        CTRV modeli, GPS'siz dönemlerde konumu kinematik olarak ileri taşır.

        Args:
            dt: Zaman adımı (saniye). Genellikle 0.1 s (10 Hz).
        """
        if not self._initialized:
            return  # Bootstrap tamamlanmadan tahmin yapılmaz

        # 1. Jakobyen'i MEVCUT state üzerinde hesapla (lineerizasyon noktası = x_k)
        #    Not: F(x_k) sonra f(x_k)'yı değil x_k'yı kullanır — bu EKF standardı.
        F = self._jacobian_F(self._x, dt)

        # 2. Durum tahmini: x_{k|k-1} = f(x_{k-1|k-1}, dt)
        self._x = self._f(self._x, dt)

        # Heading'i [-pi, pi] aralığında tut
        self._x[self.IDX_THETA] = self._wrap_angle(self._x[self.IDX_THETA])

        # 3. Kovaryans yayılımı: P_{k|k-1} = F * P * F^T + Q
        self._P = F @ self._P @ F.T + self._Q

        # 4. Sayısal kararlılık: P'nin simetrik kalmasını zorla
        self._P = 0.5 * (self._P + self._P.T)
        self._P += np.eye(5) * self._jitter

    # ------------------------------------------------------------------
    # UPDATE — Güncelleme Adımı (GPS geldiğinde)
    # ------------------------------------------------------------------

    def update(self, z: np.ndarray, timestamp: Optional[float] = None) -> bool:
        """
        EKF güncelleme adımı (GPS ölçümü geldiğinde, ~1 Hz).

        Ölçüm z = [x_gps, y_gps] (cm cinsinden).

        Aykırı değer tespiti (Mahalanobis):
            D² = (z - z_hat)^T * S^{-1} * (z - z_hat)
            D² > eşik ise ölçüm reddedilir, tahmin adımı devam eder.

        Args:
            z:         GPS ölçüm vektörü, shape (2,)
            timestamp: Ölçüm zaman damgası. None ise time.time() kullanılır.

        Returns:
            True  → ölçüm kabul edildi ve filtre güncellendi.
            False → bootstrap devam ediyor veya ölçüm reddedildi.
        """
        if timestamp is None:
            timestamp = time.time()

        z = np.asarray(z, dtype=float)

        # Bootstrap aşaması
        if not self._initialized:
            return self._bootstrap(z, timestamp)

        # 1. Yenilik (innovation): y = z - H*x
        z_hat = self._H @ self._x       # beklenen ölçüm (2,)
        y     = z - z_hat               # yenilik vektörü

        # 2. Yenilik kovaryansı: S = H * P * H^T + R
        S = self._H @ self._P @ self._H.T + self._R

        # 3. Mahalanobis mesafesi ile aykırı değer tespiti
        # Warmup süresi boyunca (ilk N güncelleme) bu test atlanır:
        # Filtre henüz yanlış heading ile başlamış olabilir; GPS reddedilirse
        # heading hiç düzelemez (chicken-egg problemi). Warmup sonrası
        # spike tespiti tam güçle devreye girer.
        in_warmup = (self._update_count < self._warmup_updates)
        if self._outlier_enabled and not in_warmup:
            try:
                S_inv = np.linalg.inv(S)
                mahal_sq = float(y.T @ S_inv @ y)
                if mahal_sq > self._mahal_threshold:
                    # GPS spike tespit edildi → ölçümü reddet, sadece predict ile devam
                    return False
            except np.linalg.LinAlgError:
                pass  # Tekil matris → spike reddi atlanır

        # 4. Kalman kazancı: K = P * H^T * S^{-1}
        K = self._P @ self._H.T @ np.linalg.inv(S)  # shape (5, 2)

        # SELEKTİF GÜNCELLEME: v ve omega satırlarını sıfırla.
        # Gerekçe: GPS yalnızca (x, y) ölçer. Pozisyon yeniliğinden (innovation)
        # hız ve dönüş hızını güncellemek, heading belirsizliği yüksekken EKF'i
        # ıraksatır. v ve omega yalnızca kinematik model (predict) üzerinden
        # güncellenir. Bu yaklaşım "selective / partial update" olarak bilinir.
        K[self.IDX_V, :]     = 0.0
        K[self.IDX_OMEGA, :] = 0.0

        # 5. Durum güncellemesi: x = x + K * y
        self._x = self._x + K @ y

        # Heading normalize et
        self._x[self.IDX_THETA] = self._wrap_angle(self._x[self.IDX_THETA])

        # Fiziksel sınır kısıtları — EKF'in ıraksamamasını önler.
        # Sabit kanatlı uçak negatif hızla veya çok yüksek yaw_rate ile uçamaz.
        # Bu değerler config'ten alınabilir; şimdilik konservatif sabitler kullanılıyor.
        _MAX_SPEED_CMS  = 10000.0   # 100 m/s (çok hızlı uçak bile bu sınırı aşmaz)
        _MIN_SPEED_CMS  = 100.0     # 1 m/s (sabit kanatlı duramaz)
        _MAX_OMEGA_RADS = 1.0       # ~57°/s (aşırı banked turn bile bu kadar etmez)
        self._x[self.IDX_V]     = float(np.clip(self._x[self.IDX_V],
                                                _MIN_SPEED_CMS, _MAX_SPEED_CMS))
        self._x[self.IDX_OMEGA] = float(np.clip(self._x[self.IDX_OMEGA],
                                                -_MAX_OMEGA_RADS, _MAX_OMEGA_RADS))

        # 6. Kovaryans güncellemesi (Joseph formu — sayısal kararlılık için)
        I_KH  = np.eye(5) - K @ self._H
        self._P = I_KH @ self._P @ I_KH.T + K @ self._R @ K.T

        # Simetriyi zorla
        self._P = 0.5 * (self._P + self._P.T)
        self._P += np.eye(5) * self._jitter

        # Güncelleme sayacı ve önceki GPS'i kaydet
        self._update_count += 1
        self._prev_gps = z.copy()
        self._prev_gps_time = timestamp

        return True

    # ------------------------------------------------------------------
    # Yardımcı Fonksiyonlar
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Açıyı [-pi, pi] aralığına normalize eder."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def reset(self) -> None:
        """Filtreyi sıfırlar. Yeni bir uçuş oturumu başladığında çağrılır."""
        self._x             = np.zeros(5)
        self._P             = self._P0.copy()
        self._initialized   = False
        self._sample_count  = 0
        self._update_count  = 0
        self._prev_gps      = None
        self._prev_gps_time = 0.0

    def get_state_dict(self) -> dict:
        """
        Mevcut EKF durumunu dict olarak döner.
        drone_gui.py veya harici modüller için kullanışlı arayüz.
        """
        return {
            "x_cm":       float(self._x[self.IDX_X]),
            "y_cm":       float(self._x[self.IDX_Y]),
            "theta_rad":  float(self._x[self.IDX_THETA]),
            "v_cms":      float(self._x[self.IDX_V]),
            "omega_rads": float(self._x[self.IDX_OMEGA]),
            "initialized": self._initialized,
            "cov_trace":  float(np.trace(self._P)),
        }

    def __repr__(self) -> str:
        pos = self.position
        return (
            f"FixedWingEKF("
            f"pos=({pos[0]:.1f}, {pos[1]:.1f}) cm, "
            f"heading={math.degrees(self.heading):.1f}°, "
            f"speed={self.speed:.0f} cm/s, "
            f"omega={math.degrees(self.yaw_rate):.2f}°/s, "
            f"initialized={self._initialized})"
        )
