"""
================================================================================
BIRLESIK TEST: Rota-Intercept (GPS) -> Gorsel Servo (Kamera) YUMUSAK DEVIR
================================================================================
Dosya: tests/python/test_rfdetr_route_handover.py

Iki mevcut testi birlestirir:
  * tests/python/test_route_intercept.py         → telemetri (GPS) tabanli rota
        ogrenme + guudum (RouteFollowGuidance + PID). Hedefe UZAKTAN yaklasir.
  * tests/python/test_rfdetr_trt_visual_servo.py → kamera (TRT+SAHI+CSRT) tabanli
        merkez kilitlenme + gorsel servo. Hedefi YAKINDAN goruntu ile takip eder.

Senaryo (mode manager):
  1. GPS_PURSUIT : Rota guudumu droneu hedefe dogru goturur. Kamera boru hatti
     bosta; hedefe olan telemetri mesafesi izlenir.
  2. (mesafe < VISION_ENABLE_CM) Kamera boru hatti CALISMAYA baslar — tespit,
     takip, merkez kilitlenme dortgeni.
  3. HANDOVER : (mesafe < HANDOVER_CM) VE kamera KILIT saglaninca, kontrol
     GPS guudumunden gorsel servoya smoothstep agirligiyla ~HANDOVER_TIME sn
     icinde YUMUSAK sekilde devredilir (komut vektorleri harmanlanir).
  4. VISUAL_SERVO : Drone artik yalnizca goruntu ile hedefi ortalar/takip eder.
     Kilit LOST_TIMEOUT boyunca kaybolursa GPS_PURSUIT'e geri duser.

Calistirma:
  # Simulasyon (donanim/simulator gerekmez — handover mantigini uctan uca gosterir)
  python tests/python/test_rfdetr_route_handover.py --sim
  python tests/python/test_rfdetr_route_handover.py --sim --headless --duration 90

  # Canli (DronesOfWar simulatoru TCP :12345 + TRT engine + ekranda oyun)
  python tests/python/test_rfdetr_route_handover.py

Not: --sim modu CUDA/TensorRT gerektirmez; sentetik kamera projeksiyonu kullanir.
     Canli mod kamera boru hattini test_rfdetr_trt_visual_servo.py'den ceker.
================================================================================
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from typing import Optional

import numpy as np

# ─── Proje yollari ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "python"))
sys.path.insert(0, os.path.dirname(__file__))

import yaml  # noqa: E402

# pyrefly: ignore [missing-import]
from navigation.route_intercept_system import RoutePlanner       # noqa: E402
from guidance.route_follow_guidance import (                     # noqa: E402
    RouteFollowGuidance, GuidePointSlewLimiter,
)
from guidance.pid_yaw import YawController                        # noqa: E402
from guidance.pid_pitch import PitchController                    # noqa: E402
from guidance.pid_throttle import ThrottleController             # noqa: E402
from vision.visual_servo import (                                # noqa: E402
    ServoConfig, LockOnManager, VisualServoController,
)
from route_viz import EllipseSimTarget                           # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
#  YAPILANDIRMA
# ═══════════════════════════════════════════════════════════════════════════════

DT_LOOP = 0.02                 # 50 Hz kontrol dongusu (config flight.dt ile ayni)

# ── Devir (handover) esikleri ─────────────────────────────────────────────────
VISION_ENABLE_CM = 12000.0     # bu mesafenin altinda kamera boru hatti calisir
HANDOVER_CM      = 8000.0      # bu mesafe + kamera KILIT → gorsel servoya devir
HANDOVER_TIME_S  = 2.0         # GPS→servo komut harmanlama suresi (smoothstep)
LOCK_STABLE_S    = 0.8         # devir baslamadan once kilit bu kadar KESINTISIZ olmali
RELOCK_GRACE_S   = 1.5         # kilit kaybinda GPS'e donmeden once bekleme (histerezis)
VISION_CONF_THRESHOLD = 0.50   # canli tespit esigi (uzak/kucuk hedef icin dusuk)

# EDINIM (acquire): kamera menzilinde ama henuz KILIT yokken, GPS guudumu hedefi
# yakalamaya calismak yerine kamerayi hedefe DOGRULTUP hedef hizina yavaslar.
# Boylece drone tasmadan hedefi kadrajda tutar → kilit hizli olusur, sonra devir.
ACQUIRE_SPEED_SCALE = 1.05     # edinimde hiz tavani = nominal_target_speed * bu
LIVE_VERBOSE = True            # canli modda periyodik teshis ciktisi (mesafe/mod/tespit)

# ── Simulasyon: hedef + kamera ────────────────────────────────────────────────
SIM_TARGET_SPEED = 1800.0      # cm/s
SIM_CENTER_X     = 20000.0
SIM_CENTER_Y     = 20000.0
SIM_A            = 15000.0
SIM_B            = 10000.0
SIM_GPS_NOISE    = 500.0
SIM_GPS_PERIOD   = 1.0         # s (yarisma: max 1 Hz)
SIM_TARGET_ALT   = 15000.0     # cm (sabit)
SIM_DRONE_ALT0   = 14000.0     # cm (baslangic — dikey servoyu gormek icin fark)
SIM_DRONE_SPEED  = 2500.0      # cm/s seyir
SIM_MAX_YAW_DPS  = 70.0        # tam yaw stick'te donus hizi
SIM_MAX_CLIMB    = 800.0       # cm/s tam throttle'da dikey hiz
SIM_CAM_W        = 1280
SIM_CAM_H        = 720

# Sim kamera projeksiyonu fiziksel dogru: hedef solda (rel>0) → piksel solda.
# Bu yonelimde yaw_sign=-1 ve throttle_sign=+1 yakinsar (bkz. _project_target).
SIM_SERVO_CFG = ServoConfig(yaw_sign=-1.0, throttle_sign=+1.0)


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


# ═══════════════════════════════════════════════════════════════════════════════
#  DEVIR (HANDOVER) DURUM MAKINESI — her iki modda ortak
# ═══════════════════════════════════════════════════════════════════════════════

class HandoverManager:
    """GPS_PURSUIT → HANDOVER → VISUAL_SERVO → (kayipsa) GPS_PURSUIT.

    `weight` 0=tam GPS, 1=tam gorsel servo. HANDOVER'da smoothstep ile rampalanir.
    Karar girisleri: hedefe telemetri mesafesi (cm) ve kamera kilit durumu.
    """

    GPS = "GPS_PURSUIT"
    HANDOVER = "HANDOVER"
    SERVO = "VISUAL_SERVO"

    def __init__(self):
        self.mode = self.GPS
        self.weight = 0.0
        self._t_in_handover = 0.0
        self._lost_t = 0.0
        self._lock_hold_t = 0.0     # kilit ne kadar suredir KESINTISIZ tutuluyor

    def vision_enabled(self, dist_cm: float) -> bool:
        """Kamera boru hattinin calismasi gereken menzil."""
        return dist_cm < VISION_ENABLE_CM

    def update(self, dist_cm: float, vision_locked: bool, dt: float) -> None:
        # Kesintisiz kilit sayaci (debounce): kilit kararli mi?
        if vision_locked:
            self._lock_hold_t += dt
            self._lost_t = 0.0
        else:
            self._lock_hold_t = 0.0
            self._lost_t += dt

        if self.mode == self.GPS:
            self.weight = 0.0
            # Devir icin kilit KARARLI olmali (LOCK_STABLE_S) — anlik kilitler
            # mod titresimine yol acmasin.
            if dist_cm < HANDOVER_CM and self._lock_hold_t >= LOCK_STABLE_S:
                self.mode = self.HANDOVER
                self._t_in_handover = 0.0

        elif self.mode == self.HANDOVER:
            # Kisa kilit kayiplarinda handover'i BIRAKMA; rampayi dondur, GPS'e
            # ancak grace suresi asilirsa don (histerezis → titresim yok).
            if self._lost_t >= RELOCK_GRACE_S:
                self.mode = self.GPS
                self.weight = 0.0
                return
            if vision_locked:
                self._t_in_handover += dt
                self.weight = smoothstep(self._t_in_handover / HANDOVER_TIME_S)
                if self.weight >= 1.0 - 1e-6:
                    self.mode = self.SERVO
                    self.weight = 1.0

        elif self.mode == self.SERVO:
            self.weight = 1.0
            if self._lost_t >= RELOCK_GRACE_S:
                self.mode = self.GPS
                self.weight = 0.0


def blend(gps_cmd, servo_cmd, w: float):
    """(thr, pit, yaw) iki komut vektorunu w agirligiyla harmanlar."""
    return tuple((1.0 - w) * g + w * s for g, s in zip(gps_cmd, servo_cmd))


# ═══════════════════════════════════════════════════════════════════════════════
#  GPS GUUDUM SARMALAYICISI — rota ogren + guide + PID (her iki modda ortak)
# ═══════════════════════════════════════════════════════════════════════════════

class GpsGuidanceStack:
    """test_route_intercept.py'nin guudum yiginini sarar: RoutePlanner (elips
    ogren) + RouteFollowGuidance (carrot) + Yaw/Pitch PID. Sentetik veya canli
    telemetriden (thr, pit, yaw) stick komutu ve istenen hiz uretir."""

    def __init__(self, config):
        self.config = config
        self.planner = RoutePlanner(buffer_size=60, min_fit_points=15,
                                    residual_threshold=1500.0)
        self.guidance = RouteFollowGuidance(config)
        self.yaw_ctrl = YawController(config)
        self.pitch_ctrl = PitchController(config)
        self.throttle_ctrl = ThrottleController(config)
        ri = config.get("route_intercept", {})
        nom = float(ri.get("nominal_target_speed_cms", 1700.0))
        self.guide_slew = GuidePointSlewLimiter(nom * 2.0)
        self._last_yaw = 0.0

    def add_gps(self, t, tx, ty):
        self.planner.add_observation(t, tx, ty)

    def step(self, t, dx, dy, dz, heading_deg, v_fwd, tx, ty, locked_alt, dt,
             acquire=False):
        """Bir GPS-guudum adimi. Doner: (thr, pit, yaw, v_des, guide_xy, dist2d).

        acquire=True: rota carrot'u yerine kamerayi HAM hedefe dogrultur ve
        hiz tavanini hedef hizina indirir (tasmayi onleyip kadrajda tutar)."""
        if acquire:
            # Kamerayi hedefe dogrult; carrot devre disi
            guide_x, guide_y = tx, ty
            ring_v = None
        else:
            guide_x, guide_y = tx, ty
            ring_v = None
            if self.planner.route.fitted:
                guide_x, guide_y, ring_v = self.guidance.guide(
                    self.planner.route, dx, dy, tx, ty, dt=dt)
        guide_x, guide_y = self.guide_slew.update(guide_x, guide_y, dt)

        ldx, ldy = guide_x - dx, guide_y - dy
        ldz = locked_alt - dz
        dist2d = math.hypot(ldx, ldy)

        yaw_err, yaw_cmd = self.yaw_ctrl.calculate(ldx, ldy, heading_deg, dt)
        v_des = self.guidance.desired_speed(ring_v, dist2d, waiting=False, data_stale=False)
        if acquire:
            v_des = min(v_des, self.guidance.nominal_target_speed * ACQUIRE_SPEED_SCALE)

        target_elev = math.degrees(math.atan2(ldz, max(dist2d, 1.0)))
        _, pitch_cmd = self.pitch_ctrl.calculate(
            state="TAKIP", target_speed_cmd=v_des, drone_speed=v_fwd,
            target_elevation=target_elev, drone_pitch=0.0,
            yaw_error=yaw_err, dt=dt)
        thr_cmd = self.throttle_ctrl.calculate(ldz, dt)

        thr = max(-1.0, min(1.0, thr_cmd))
        pit = -max(-1.0, min(1.0, pitch_cmd))     # sim_bridge isaret duzeni
        yaw = max(-1.0, min(1.0, yaw_cmd))
        return thr, pit, yaw, v_des, (guide_x, guide_y), dist2d


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULASYON MODU
# ═══════════════════════════════════════════════════════════════════════════════

def _project_target(dx, dy, dz, heading_deg, tx, ty, tz, cfg: ServoConfig):
    """Sentetik ileri-kamera projeksiyonu.

    Hedefin drone burnuna gore yatay/dikey acisini piksel koordinatina cevirir.
    Hedef FOV disindaysa None doner (kamera goremiyor).
    Fiziksel dogru: hedef SOLDA (rel>0) → piksel SOLDA (< merkez).
    """
    bearing = math.degrees(math.atan2(ty - dy, tx - dx))
    rel_h = wrap180(bearing - heading_deg)              # + = hedef solda
    if abs(rel_h) > cfg.camera_fov_h_deg / 2.0:
        return None

    fov_h = cfg.camera_fov_h_deg
    half_h = math.radians(fov_h / 2.0)
    fov_v = math.degrees(2.0 * math.atan(math.tan(half_h) * SIM_CAM_H / SIM_CAM_W))

    dist2d = max(math.hypot(tx - dx, ty - dy), 1.0)
    rel_v = math.degrees(math.atan2(tz - dz, dist2d))   # + = hedef yukarida
    if abs(rel_v) > fov_v / 2.0:
        return None

    px = SIM_CAM_W / 2.0 - (rel_h / (fov_h / 2.0)) * (SIM_CAM_W / 2.0)
    py = SIM_CAM_H / 2.0 - (rel_v / (fov_v / 2.0)) * (SIM_CAM_H / 2.0)
    return px, py


class HandoverSimulator:
    """Elips hedefi + kinematik drone + sentetik kamera ile GPS→servo devir
    senaryosunu uctan uca kosar. Donanim/simulator gerektirmez."""

    def __init__(self, headless=False, duration=90.0, log_dir=None):
        self.headless = headless
        self.duration = duration or 90.0
        self.log_dir = log_dir or os.getcwd()

        cfg_path = os.path.join(PROJECT_ROOT, "config", "pid_params.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.gps = GpsGuidanceStack(self.config)
        self.lock = LockOnManager(SIM_CAM_W, SIM_CAM_H, SIM_SERVO_CFG)
        self.servo = VisualServoController(SIM_CAM_W, SIM_CAM_H, SIM_SERVO_CFG)
        self.handover = HandoverManager()
        self.target = EllipseSimTarget(SIM_CENTER_X, SIM_CENTER_Y, SIM_A, SIM_B,
                                       SIM_TARGET_SPEED, SIM_GPS_NOISE)

    def run(self):
        print("=" * 64)
        print("  BIRLESIK GPS→GORSEL SERVO DEVIR SIMULASYONU")
        print(f"  Hedef {SIM_TARGET_SPEED/100:.1f} m/s | Drone {SIM_DRONE_SPEED/100:.1f} m/s")
        print(f"  Kamera menzili {VISION_ENABLE_CM/100:.0f} m | Devir {HANDOVER_CM/100:.0f} m + kilit")
        print("=" * 64)

        # Kinematik drone durumu
        dx, dy, dz = 0.0, 0.0, SIM_DRONE_ALT0
        # Baslangicta hedefe dogru bak
        heading = math.degrees(math.atan2(SIM_CENTER_Y - dy, SIM_CENTER_X - dx))
        speed = SIM_DRONE_SPEED

        t = 0.0
        last_gps_t = -1e9
        step = 0
        log = []
        mode_changes = []
        prev_mode = self.handover.mode

        while t < self.duration:
            tx_real, ty_real = self.target.truth(t)
            tz_real = SIM_TARGET_ALT

            # 1 Hz gurultulu GPS → rota ogren
            if t - last_gps_t >= SIM_GPS_PERIOD:
                txn, tyn = self.target.measure(t)
                last_gps_t = t
                self.gps.add_gps(t, txn, tyn)
                gps_tx, gps_ty = txn, tyn
            # (guudum en son 1 Hz gps'i kullanir; stale-hold gercekci)

            dist_real = math.hypot(tx_real - dx, ty_real - dy)

            # ── Kamera boru hatti (yalnizca menzil icinde) ────────────────────
            vision_locked = False
            target_px = None
            if self.handover.vision_enabled(dist_real):
                target_px = _project_target(dx, dy, dz, heading,
                                            tx_real, ty_real, tz_real, SIM_SERVO_CFG)
                if target_px is not None:
                    # Tek sentetik tespit → kilit durum makinesi
                    px, py = target_px
                    box = np.array([[px - 30, py - 20, px + 30, py + 20, 0.9, 0, 1]])
                    self.lock.update(box)
                    vision_locked = self.lock.locked
                else:
                    self.lock.reset()

            # ── Mode manager ──────────────────────────────────────────────────
            self.handover.update(dist_real, vision_locked, DT_LOOP)
            if self.handover.mode != prev_mode:
                mode_changes.append((t, prev_mode, self.handover.mode))
                print(f"  [t={t:5.1f}s] MOD: {prev_mode} → {self.handover.mode} "
                      f"(mesafe {dist_real/100:.1f} m, kilit={vision_locked})")
                prev_mode = self.handover.mode

            # ── GPS guudum komutu ─────────────────────────────────────────────
            hd = math.radians(heading)
            v_fwd = speed
            gps_thr, gps_pit, gps_yaw, v_des, guide_xy, dist2d = self.gps.step(
                t, dx, dy, dz, heading, v_fwd,
                gps_tx, gps_ty, SIM_TARGET_ALT, DT_LOOP)

            # ── Gorsel servo komutu ───────────────────────────────────────────
            if self.handover.weight > 0.0:
                s_thr, s_pit, s_yaw, alive = self.servo.update(target_px, DT_LOOP)
            else:
                s_thr = s_pit = s_yaw = 0.0
                self.servo.reset()

            # ── Harmanla ──────────────────────────────────────────────────────
            w = self.handover.weight
            thr, pit, yaw = blend((gps_thr, gps_pit, gps_yaw),
                                  (s_thr, s_pit, s_yaw), w)

            # ── Kinematik model: stick → hareket ──────────────────────────────
            heading += yaw * SIM_MAX_YAW_DPS * DT_LOOP
            # GPS fazinda hiz guudumden; servo fazinda hedef hizini takip et
            v_cmd = (1.0 - w) * v_des + w * SIM_TARGET_SPEED
            speed += max(-400.0, min(400.0, v_cmd - speed))   # basit ivme siniri
            dz += thr * SIM_MAX_CLIMB * DT_LOOP
            hd = math.radians(heading)
            dx += speed * math.cos(hd) * DT_LOOP
            dy += speed * math.sin(hd) * DT_LOOP

            # Piksel hatasi (rapor)
            if target_px is not None:
                ex = target_px[0] - SIM_CAM_W / 2.0
                ey = target_px[1] - SIM_CAM_H / 2.0
            else:
                ex = ey = np.nan

            log.append((t, dx, dy, dz, tx_real, ty_real, dist_real, dist2d,
                        heading, speed, w, thr, pit, yaw, ex, ey,
                        {"GPS_PURSUIT": 0, "HANDOVER": 1, "VISUAL_SERVO": 2}[self.handover.mode]))
            step += 1
            t += DT_LOOP

        self._report(log, mode_changes)

    def _report(self, log, mode_changes):
        arr = np.array([[r[i] for i in range(16)] + [r[16]] for r in log], dtype=float)
        cols = ["t", "dx", "dy", "dz", "tx", "ty", "dist", "dist2d",
                "heading", "speed", "w", "thr", "pit", "yaw", "ex", "ey", "mode"]
        c = {n: arr[:, i] for i, n in enumerate(cols)}
        t = c["t"]

        servo_mask = c["mode"] >= 2
        min_dist = float(np.min(c["dist"]))
        print("\n  " + "=" * 56)
        print("  DEVIR SIMULASYONU RAPORU")
        print(f"    Sure                  : {t[-1]:.1f} s ({len(t)} adim)")
        print(f"    Mod gecisleri         : {len(mode_changes)}")
        for tt, a, b in mode_changes:
            print(f"      t={tt:5.1f}s  {a} → {b}")
        print(f"    En yakin gercek mesafe: {min_dist:.0f} cm ({min_dist/100:.1f} m)")
        if servo_mask.any():
            seg = servo_mask & np.isfinite(c["ex"])
            if seg.any():
                px_rms = float(np.sqrt(np.mean(c["ex"][seg] ** 2)))
                py_rms = float(np.sqrt(np.mean(c["ey"][seg] ** 2)))
                print(f"    Servo fazi piksel hata: x_RMS {px_rms:5.0f} / "
                      f"y_RMS {py_rms:5.0f} px (merkez=0)")
                print(f"    Servo fazi mesafe      : ort {np.mean(c['dist'][servo_mask]):.0f}"
                      f" / son {c['dist'][-1]:.0f} cm")
            dyaw = np.abs(np.diff(c["yaw"]))
            print(f"    Yaw komut degisim ort : {np.mean(dyaw):.4f}/adim "
                  f"(yumusaklik; dusuk=iyi)")
        else:
            print("    [UYARI] Servo fazina hic girilmedi — esikleri gozden gecir.")

        if not self.headless:
            self._plot(c, mode_changes)
        else:
            self._plot(c, mode_changes, save_only=True)

    def _plot(self, c, mode_changes, save_only=False):
        import matplotlib
        if save_only:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        fig.suptitle("Birlesik GPS → Gorsel Servo Devir Simulasyonu", fontweight="bold")

        ax = axes[0, 0]
        ax.plot(c["tx"], c["ty"], ".", ms=2, color="tab:red", alpha=0.5, label="Hedef")
        # Modlara gore droneu renklendir
        for lo, hi, col, lab in [(0, 1, "tab:blue", "GPS"),
                                 (1, 2, "tab:orange", "Handover"),
                                 (2, 3, "tab:green", "Servo")]:
            m = (c["mode"] >= lo) & (c["mode"] < hi)
            ax.plot(c["dx"][m], c["dy"][m], ".", ms=3, color=col, label=lab)
        ax.set_title("XY Yorunge (moda gore renk)"); ax.axis("equal")
        ax.legend(); ax.grid(True, alpha=0.4)

        ax = axes[0, 1]
        ax.plot(c["t"], c["dist"] / 100.0, color="tab:blue", label="Gercek mesafe (m)")
        ax.axhline(VISION_ENABLE_CM / 100, ls="--", color="gray", label="Kamera menzili")
        ax.axhline(HANDOVER_CM / 100, ls=":", color="tab:orange", label="Devir esigi")
        ax.set_title("Hedef Mesafesi"); ax.set_xlabel("t (s)"); ax.set_ylabel("m")
        ax.legend(); ax.grid(True, alpha=0.4)

        ax = axes[1, 0]
        ax.plot(c["t"], c["w"], color="tab:purple", label="Servo agirligi w")
        ax.plot(c["t"], c["mode"] / 2.0, color="tab:green", lw=0.8, alpha=0.6, label="Mod/2")
        ax.set_title("Devir Agirligi (0=GPS, 1=Servo)")
        ax.set_xlabel("t (s)"); ax.legend(); ax.grid(True, alpha=0.4)

        ax = axes[1, 1]
        ax.plot(c["t"], c["ex"], color="tab:red", lw=0.8, label="piksel hata x")
        ax.plot(c["t"], c["ey"], color="tab:blue", lw=0.8, label="piksel hata y")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title("Kamera Piksel Hatasi (merkez=0)")
        ax.set_xlabel("t (s)"); ax.set_ylabel("px"); ax.legend(); ax.grid(True, alpha=0.4)

        fig.tight_layout()
        png = os.path.join(self.log_dir, "route_handover_sim.png")
        os.makedirs(self.log_dir, exist_ok=True)
        fig.savefig(png, dpi=110)
        print(f"    Rapor PNG             : {png}")
        if not save_only:
            plt.show()
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  CANLI MOD — Kamera boru hatti (thread) + 50 Hz kontrol dongusu
# ═══════════════════════════════════════════════════════════════════════════════

class VisionWorker(threading.Thread):
    """Arka planda TRT+SAHI+CSRT tespit/takip + merkez kilitlenme kosar.
    Kontrol dongusu icin en guncel (locked, target_px) durumunu yayinlar.
    Agir kamera boru hatti test_rfdetr_trt_visual_servo.py'den cekilir."""

    def __init__(self, servo_cfg: ServoConfig, headless: bool):
        super().__init__(daemon=True)
        import test_rfdetr_trt_visual_servo as vs   # lazy: CUDA yalnizca canli modda
        self.vs = vs
        # Edinim menzilinde hedefi daha sik yakalamak icin tespit esigini dusur
        # (uzak/kucuk hedef 0.70'te seyrek geliyordu). Kilit + histerezis
        # yanlis pozitifleri zaten filtreler.
        vs.rt.CONF_THRESHOLD = VISION_CONF_THRESHOLD
        self.headless = headless
        self._enabled = False
        self._running = True
        self._lock = threading.Lock()
        # (locked, target_px, err_px, n_tracked)
        self._state = (False, None, (0.0, 0.0), 0)

        self.inferencer = vs.rt.TRTInferencer(vs.ENGINE_PATH)
        self.inferencer.warmup(n=3)
        self.capture = vs.rt.ScreenCapture(vs.MONITOR_INDEX, vs.RESIZE_SCALE)
        while self.capture.get() is None:
            time.sleep(0.005)
        f = self.capture.get()
        self.h, self.w = f.shape[:2]
        self.lockmgr = vs.LockOnManager(self.w, self.h, servo_cfg)
        self.sort = vs.Sort(max_age=vs.SORT_MAX_AGE, min_hits=vs.SORT_MIN_HITS,
                            iou_threshold=vs.SORT_IOU_THRESH)

    def set_enabled(self, on: bool):
        self._enabled = on

    def get_state(self):
        with self._lock:
            return self._state

    def stop(self):
        self._running = False

    def run(self):
        vs = self.vs
        in_track = False
        phase_start = time.perf_counter()
        prev_raw = None
        was_enabled = False
        while self._running:
            frame = self.capture.get()
            if not self._enabled:
                # Menzil disi: BAYAT kilit yayinlama — aksi halde handover
                # donmus pikseli gercek kilit sanip droneu savuruyordu.
                if was_enabled:
                    self.lockmgr.reset()
                    in_track = False
                    with self._lock:
                        self._state = (False, None, (0.0, 0.0), 0)
                    was_enabled = False
                time.sleep(0.005)
                continue
            was_enabled = True
            if frame is None or frame is prev_raw:
                time.sleep(0.002)
                continue
            prev_raw = frame
            elapsed = time.perf_counter() - phase_start

            if not in_track:
                xyxy, confs, cls = vs.rt.sahi_predict(self.inferencer, frame)
                dets = (np.column_stack([xyxy, confs, cls])
                        if len(confs) else np.empty((0, 6)))
                tracked = vs._quiet_sort_update(self.sort, frame, dets)
                if elapsed >= vs.DETECT_DURATION:
                    if len(tracked):
                        vs.prune_to_lock_target(self.sort, self.lockmgr)
                        in_track = True
                    phase_start = time.perf_counter()
            else:
                tracked = vs._quiet_sort_update(self.sort, frame, np.empty((0, 6)))
                if len(tracked) == 0 or elapsed >= vs.TRACK_DURATION:
                    in_track = False
                    phase_start = time.perf_counter()

            row = self.lockmgr.update(tracked)
            if row is not None:
                px = ((row[0] + row[2]) / 2.0, (row[1] + row[3]) / 2.0)
            else:
                px = None
            err = ((px[0] - self.w / 2.0, px[1] - self.h / 2.0)
                   if px is not None else (0.0, 0.0))
            with self._lock:
                self._state = (self.lockmgr.locked, px, err, len(tracked))

            if not self.headless:
                import cv2
                disp = frame.copy()
                vs.draw_lock_reticle(disp, self.lockmgr)
                cv2.imshow("Kamera Boru Hatti (kilit)", disp)
                cv2.waitKey(1)


class LiveHandoverRunner:
    """Canli mod: drone_sdk telemetri + rota guudumu ile yaklas, yakinda kamera
    kilidiyle gorsel servoya yumusak devret."""

    def __init__(self, headless=False, duration=None):
        self.headless = headless
        self.duration = duration
        cfg_path = os.path.join(PROJECT_ROOT, "config", "pid_params.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        # Canli kilit: uzak/kucuk hedef icin biraz daha kolay edinim (kutu 0.22→
        # 0.30, onay 8→5 kare). Isaretler (yaw/throttle) sahada kalibre edilir.
        self.servo_cfg = ServoConfig(lock_box_frac=0.30, lock_confirm_frames=5)
        self.gps = GpsGuidanceStack(self.config)
        self.handover = HandoverManager()

    def run(self):
        import drone_sdk as drone
        print("  drone_sdk'ya baglaniliyor (127.0.0.1:12345) ...")
        for _ in range(30):
            if drone.connect():
                break
            print("  ✗ Baglanti yok — 2 sn sonra tekrar ...")
            time.sleep(2.0)
        if not drone.is_connected():
            print("  [HATA] Simulator baglantisi kurulamadi.")
            return
        print("  ✓ Baglandi. Kamera boru hatti baslatiliyor (TRT engine yukleniyor)...")

        try:
            vision = VisionWorker(self.servo_cfg, self.headless)
        except Exception as e:
            print(f"  [HATA] Kamera boru hatti baslatilamadi: {e}")
            drone.disconnect()
            return
        servo = VisualServoController(vision.w, vision.h, self.servo_cfg)
        vision.start()

        print("  ✓ Hazir. GPS_PURSUIT ile yaklasiliyor. CTRL+C ile durdur.\n")
        drone.set_arm(True)

        t0 = time.time()
        prev_loop = t0
        last_gps = -1e9
        prev_mode = self.handover.mode
        prev_heading = None
        step = 0

        try:
            while True:
                now = time.time()
                t_now = now - t0
                dt = max(0.005, min(0.1, now - prev_loop))
                prev_loop = now

                td = drone.get_telemetry()
                d_pos = td["drone"]["position"]; d_rot = td["drone"]["rotation"]
                d_vel = td["drone"]["velocity"]; tar = td["target"]["position"]
                dx, dy, dz = d_pos
                tx, ty, tz = tar
                heading = d_rot[2]

                if dx == 0.0 and dy == 0.0 and tx == 0.0 and ty == 0.0:
                    time.sleep(DT_LOOP); continue

                if t_now - last_gps >= SIM_GPS_PERIOD:
                    self.gps.add_gps(t_now, tx, ty)
                    last_gps = t_now

                dist = math.hypot(tx - dx, ty - dy)

                # Kamerayi yalnizca menzil icinde calistir
                vis_on = self.handover.vision_enabled(dist)
                vision.set_enabled(vis_on)
                vlocked, vpx, _, vn = vision.get_state()
                if not vis_on:
                    vlocked, vpx = False, None   # menzil disi = kilit yok

                self.handover.update(dist, vlocked, dt)
                if self.handover.mode != prev_mode:
                    print(f"  [t={t_now:5.1f}s] MOD: {prev_mode} → {self.handover.mode} "
                          f"(mesafe {dist/100:.1f} m, kilit={vlocked})", flush=True)
                    prev_mode = self.handover.mode

                # EDINIM: menzilde ama kilit yok ve hala GPS modunda → kamerayi
                # hedefe dogrultup yavasla (tasma yerine kadrajda tut).
                acquiring = vis_on and not vlocked and self.handover.mode == HandoverManager.GPS

                # Periyodik teshis (her ~2 s)
                step += 1
                if LIVE_VERBOSE and step % 100 == 0:
                    print(f"  [t={t_now:5.1f}s] mesafe={dist/100:6.1f}m mod={self.handover.mode:<12s} "
                          f"kam={'ON' if vis_on else 'off'} edinim={'E' if acquiring else '-'} "
                          f"tespit={vn} kilit={vlocked} "
                          f"px={'-' if vpx is None else '(%d,%d)'%(vpx[0],vpx[1])} | "
                          f"drone=({dx:.0f},{dy:.0f},{dz:.0f}) hedef=({tx:.0f},{ty:.0f},{tz:.0f}) "
                          f"route_fit={self.gps.planner.route.fitted}", flush=True)

                # GPS guudum
                hd = math.radians(heading)
                v_fwd = d_vel[0] * math.cos(hd) + d_vel[1] * math.sin(hd)
                gps_thr, gps_pit, gps_yaw, _, _, _ = self.gps.step(
                    t_now, dx, dy, dz, heading, v_fwd, tx, ty, tz, dt,
                    acquire=acquiring)

                # Gorsel servo — YALNIZCA yaw (yonelim/bearing) kanalini surer.
                # Hiz (pitch) ve irtifa (throttle) DAIMA GPS guudumunden gelir:
                #  * gorsel dikey servo canli testte irtifayi savurdu (8.5→33 km),
                #  * gorsel servonun ileri pitch yetkisi 1800 cm/s hedefi
                #    kovalamaya yetmiyordu (drone geride kalip kadraji kaybediyor).
                # Boylece GPS pace/irtifayi tutar, KAMERA hassas yonelimi surer.
                w = self.handover.weight
                if w > 0.0:
                    _, _, s_yaw, _ = servo.update(vpx if vlocked else None, dt)
                else:
                    s_yaw = 0.0
                    servo.reset()

                yaw = (1.0 - w) * gps_yaw + w * s_yaw
                thr, pit = gps_thr, gps_pit
                drone.set_control_surfaces(
                    max(-1, min(1, thr)), max(-1, min(1, pit)), 0.0,
                    max(-1, min(1, yaw)), True)

                # Servo/handover fazinda yonelim teshisi: px merkeze yaklasiyor
                # mu? (yaklasmiyorsa yaw_sign ters demektir)
                if LIVE_VERBOSE and step % 50 == 0 and self.handover.mode != HandoverManager.GPS and vpx is not None:
                    ex = vpx[0] - vision.w / 2.0
                    print(f"    [servo t={t_now:5.1f}s w={w:.2f}] px_err_x={ex:+6.0f} "
                          f"s_yaw={s_yaw:+.3f} yaw={yaw:+.3f} mesafe={dist/100:.1f}m", flush=True)

                if self.duration is not None and t_now >= self.duration:
                    print(f"\n  Sure doldu ({self.duration:.0f} s).")
                    break

                slp = DT_LOOP - (time.time() - now)
                if slp > 0:
                    time.sleep(slp)

        except KeyboardInterrupt:
            print("\n  Durduruldu.")
        finally:
            # Once komutlari notrle, sonra kamera thread'ini durdurup BEKLE:
            # worker CUDA context'i tutuyor; join etmeden cikmak pycuda temizlik
            # cakismasi ("FATAL: exception not rethrown") uretiyor.
            try:
                drone.set_control_surfaces(0.0, 0.0, 0.0, 0.0, True)
            except Exception:
                pass
            vision.stop()
            vision.join(timeout=3.0)
            drone.disconnect()
            if not self.headless:
                try:
                    import cv2
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            # pycuda context'i daemon thread'de olusturuldugu icin normal
            # yorumlayici cikisi "FATAL: exception not rethrown" ile abort
            # ediyor. Rapor/temizlik bitti — sert cikisla bu kozmetik cakismayi
            # atla (SIGABRT/exit 134 yerine temiz exit 0).
            sys.stdout.flush()
            os._exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
#  GIRIS
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Rota-intercept (GPS) + gorsel servo (kamera) yumusak devir testi")
    ap.add_argument("--sim", action="store_true",
                    help="Simulasyon (donanim gerekmez; sentetik kamera).")
    ap.add_argument("--headless", action="store_true",
                    help="GUI acmadan kosar; rapor PNG uretir.")
    ap.add_argument("--duration", type=float, default=None,
                    help="Kosu suresi (s).")
    ap.add_argument("--log-dir", default=None, help="Rapor/PNG cikti dizini.")
    args = ap.parse_args()

    if args.sim:
        np.random.seed(42)
        HandoverSimulator(headless=args.headless,
                          duration=args.duration,
                          log_dir=args.log_dir).run()
    else:
        LiveHandoverRunner(headless=args.headless,
                           duration=args.duration).run()


if __name__ == "__main__":
    main()
