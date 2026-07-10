"""
GORSEL SERVO + MERKEZ KILITLENME — Yeniden Kullanilabilir Cekirdek
==================================================================
Bu modul, ekran/kamera goruntusundeki bir hedefin merkez kilitlenme
dortgeni durum makinesini (`LockOnManager`) ve piksel hatasindan yumusak
stick komutu ureten gorsel servolamayi (`VisualServoController`) icerir.
CUDA/TensorRT'ye BAGIMLI DEGILDIR — sadece numpy + math. Boylece hem canli
TRT testinden hem de simulasyon/handover testinden import edilebilir.

Tasarim felsefesi pursuit v2 / route-follow ile ayni: ham piksel hatasi
LPF ile yumusatilir, PD (turev filtreli hatadan) + smoothstep devreye-giris
rampasi + kanal basina slew limiti uygulanir; boylece kilit aninda komut
sicramasi ve tespit gurultusunden bang-bang olusmaz.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  YAPILANDIRMA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ServoConfig:
    """Kilitlenme + gorsel servo parametreleri. Cagiran test kendi
    sabitlerinden bir ornek olusturur; degerler tek yerde kalir."""

    # Kilitlenme dortgeni
    lock_box_frac: float       = 0.22   # dortgen kenari = min(w, h) * frac
    lock_hyst_scale: float     = 1.6    # kilidi dusurme dortgeni carpani
    lock_confirm_frames: int   = 8      # kilit onayi icin ardisik kare
    lock_release_frames: int   = 12     # kilidi dusurmek icin ardisik "disarida" kare

    # Kamera geometrisi (SDK: 125 derece yatay FOV, 25 derece yukari tilt)
    camera_fov_h_deg: float    = 125.0

    # Servo kazanclari
    kp_yaw: float              = 0.022  # stick / derece
    kd_yaw: float              = 0.012  # stick / (derece/s) — filtreli hata turevi
    kp_vert: float             = 0.020  # throttle / derece
    pitch_max: float           = 0.25   # hedef merkezdeyken ileri yaklasma pitch'i
    yaw_max: float             = 0.60
    thr_max: float             = 0.50
    err_tau_s: float           = 0.12   # hata LPF zaman sabiti (s)
    ramp_s: float              = 1.5    # devreye giris rampasi (s)
    slew: dict                 = field(default_factory=lambda: {
        "yaw": 2.0, "pitch": 0.8, "throttle": 1.5})  # stick/s
    lost_timeout_s: float      = 1.5    # hedef kayipken servo kapanma suresi (s)
    yaw_sign: float            = +1.0   # isaret ters gelirse -1
    throttle_sign: float       = +1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  MERKEZ KILITLENME YONETICISI
# ═══════════════════════════════════════════════════════════════════════════════

class LockOnManager:
    """Merkez kilitlenme dortgeni durum makinesi.

    Hedef secimi track ID'ye yapiskandir: mevcut secili hedef takipte kaldigi
    surece merkeze daha yakin baska hedefe atlamaz (kilit kararliligi).
    Kilit onayi/dusmesi kare sayaclariyla, dusme histerezisli dortgenle yapilir.
    """

    def __init__(self, frame_w: int, frame_h: int, cfg: Optional[ServoConfig] = None):
        self.cfg = cfg or ServoConfig()
        self.cx = frame_w // 2
        self.cy = frame_h // 2
        half = int(min(frame_w, frame_h) * self.cfg.lock_box_frac / 2)
        self.box = (self.cx - half, self.cy - half, self.cx + half, self.cy + half)
        hh = int(half * self.cfg.lock_hyst_scale)
        self.hyst_box = (self.cx - hh, self.cy - hh, self.cx + hh, self.cy + hh)

        self.target_id: Optional[int] = None
        self.target_row: Optional[np.ndarray] = None
        self.locked = False
        self._in_frames = 0
        self._out_frames = 0

    @staticmethod
    def _center(row: np.ndarray) -> tuple[float, float]:
        return (row[0] + row[2]) / 2.0, (row[1] + row[3]) / 2.0

    def _inside(self, cx: float, cy: float, box: tuple) -> bool:
        return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]

    def reset(self) -> None:
        self.target_id = None
        self.target_row = None
        self.locked = False
        self._in_frames = 0
        self._out_frames = 0

    def update(self, tracked: np.ndarray) -> Optional[np.ndarray]:
        """Takip listesinden hedef secer, kilit durumunu gunceller.

        tracked format: [[x1,y1,x2,y2,conf,class,track_id], ...] (N,7)
        Secili hedef satirini (veya None) doner.
        """
        if len(tracked) == 0:
            self.reset()
            return None

        # Yapiskan secim: mevcut ID hala takipteyse onu koru
        row = None
        if self.target_id is not None:
            ids = tracked[:, 6].astype(int)
            hits = np.where(ids == self.target_id)[0]
            if len(hits):
                row = tracked[hits[0]]

        if row is None:
            # Merkeze en yakin hedefi sec
            cxs = (tracked[:, 0] + tracked[:, 2]) / 2.0
            cys = (tracked[:, 1] + tracked[:, 3]) / 2.0
            d2 = (cxs - self.cx) ** 2 + (cys - self.cy) ** 2
            row = tracked[int(np.argmin(d2))]
            self.target_id = int(row[6])
            self.locked = False
            self._in_frames = 0
            self._out_frames = 0

        self.target_row = row
        tx, ty = self._center(row)

        if not self.locked:
            if self._inside(tx, ty, self.box):
                self._in_frames += 1
                if self._in_frames >= self.cfg.lock_confirm_frames:
                    self.locked = True
                    self._out_frames = 0
            else:
                self._in_frames = 0
        else:
            if self._inside(tx, ty, self.hyst_box):
                self._out_frames = 0
            else:
                self._out_frames += 1
                if self._out_frames >= self.cfg.lock_release_frames:
                    self.locked = False
                    self._in_frames = 0

        return row


# ═══════════════════════════════════════════════════════════════════════════════
#  GORSEL SERVO KONTROLCUSU
# ═══════════════════════════════════════════════════════════════════════════════

class VisualServoController:
    """Piksel hatasindan yumusak (throttle, pitch, yaw) stick komutlari uretir.

    Yumusaklik zinciri:
      1. Piksel hatasi FOV ile dereceye cevrilir (olcek kamera geometrisinden).
      2. Hata LPF (err_tau_s) — ham tespit gurultusunu yumusatir.
      3. PD: P filtreli hataya, D filtreli hatanin turevine (bang-bang onler).
      4. Devreye giris rampasi (smoothstep, ramp_s) — kilit aninda komut sicramaz.
      5. Kanal basina slew limiti — stick degisim hizi sinirlanir.
      6. Pitch, yaw hizalanmasiyla olceklenir (cos) — hedefe donmeden ileri
         atilmaz (pid_pitch.py'deki yaw_scale prensibi).
    Hedef kayipsa komutlar LPF ile sifira sondurulur.
    """

    def __init__(self, frame_w: int, frame_h: int, cfg: Optional[ServoConfig] = None):
        self.cfg = cfg or ServoConfig()
        self.w = frame_w
        self.h = frame_h
        self.fov_h = self.cfg.camera_fov_h_deg
        # Dikey FOV: yatay FOV + cerceve orani (pinhole)
        half_h = math.radians(self.fov_h / 2.0)
        self.fov_v = math.degrees(2.0 * math.atan(math.tan(half_h) * frame_h / frame_w))

        self._ax_f = 0.0       # filtreli yatay aci hatasi (derece, + = hedef sagda)
        self._ay_f = 0.0       # filtreli dikey aci hatasi (derece, + = hedef yukarida)
        self._ax_prev = 0.0
        self._engage_t = 0.0
        self._lost_t = 0.0
        self._cmd = {"yaw": 0.0, "pitch": 0.0, "throttle": 0.0}

    def reset(self) -> None:
        self._ax_f = self._ay_f = self._ax_prev = 0.0
        self._engage_t = 0.0
        self._lost_t = 0.0
        self._cmd = {"yaw": 0.0, "pitch": 0.0, "throttle": 0.0}

    def _slew(self, name: str, desired: float, dt: float) -> float:
        max_step = self.cfg.slew[name] * dt
        cur = self._cmd[name]
        cur += max(-max_step, min(max_step, desired - cur))
        self._cmd[name] = cur
        return cur

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def update(
        self,
        target_px: Optional[tuple[float, float]],
        dt: float,
    ) -> tuple[float, float, float, bool]:
        """Bir kontrol adimi.

        Args:
            target_px: kilitli hedef merkezi (px) veya None (hedef kayip)
            dt: zaman adimi (s)
        Returns:
            (throttle, pitch, yaw, alive) — alive=False ise servo kapatilmali
            (hedef lost_timeout_s boyunca kayip).
        """
        c = self.cfg
        dt = max(dt, 1e-3)
        self._engage_t += dt
        ramp = self._smoothstep(self._engage_t / c.ramp_s)

        if target_px is None:
            # Hedef kayip: komutlari sifira sondur, zaman asimini isle
            self._lost_t += dt
            yaw = self._slew("yaw", 0.0, dt)
            pit = self._slew("pitch", 0.0, dt)
            thr = self._slew("throttle", 0.0, dt)
            return thr, pit, yaw, self._lost_t < c.lost_timeout_s

        self._lost_t = 0.0
        tx, ty = target_px

        # 1) piksel -> aci (derece). ay: goruntu y'si asagi, aci yukari pozitif.
        ax = (tx - self.w / 2.0) / (self.w / 2.0) * (self.fov_h / 2.0)
        ay = -(ty - self.h / 2.0) / (self.h / 2.0) * (self.fov_v / 2.0)

        # 2) LPF
        alpha = 1.0 - math.exp(-dt / c.err_tau_s)
        self._ax_f += alpha * (ax - self._ax_f)
        self._ay_f += alpha * (ay - self._ay_f)

        # 3) PD (yaw): turev filtreli hatadan — ham tespit gurultusu D'yi patlatmaz
        dax = (self._ax_f - self._ax_prev) / dt
        self._ax_prev = self._ax_f

        yaw_des = c.yaw_sign * (c.kp_yaw * self._ax_f + c.kd_yaw * dax)
        yaw_des = max(-c.yaw_max, min(c.yaw_max, yaw_des)) * ramp

        # Dikey: hedef yukarida (+ay) -> tirman (+throttle). SDK: 0=hover.
        thr_des = c.throttle_sign * c.kp_vert * self._ay_f
        thr_des = max(-c.thr_max, min(c.thr_max, thr_des)) * ramp

        # Ileri yaklasma: yalnizca hedefe donukken (cos olcekleme, taban 0)
        align = max(0.0, math.cos(math.radians(min(abs(self._ax_f) * 3.0, 90.0))))
        pit_des = c.pitch_max * align * ramp

        # 5) slew limit
        yaw = self._slew("yaw", yaw_des, dt)
        pit = self._slew("pitch", pit_des, dt)
        thr = self._slew("throttle", thr_des, dt)
        return thr, pit, yaw, True

    @property
    def filtered_error_deg(self) -> tuple[float, float]:
        return self._ax_f, self._ay_f
