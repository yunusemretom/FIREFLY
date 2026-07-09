"""
================================================================================
TEST: Route Intercept System — Canlı SDK & İnteraktif Yakalama Simülasyonu
================================================================================
Dosya: tests/python/test_route_intercept.py

Çalıştırma:
    python tests/python/test_route_intercept.py [--sim] [--headless --duration N]

Mod seçimi:
    --sim yoksa: drone_sdk üzerinden canlı telemetri okunur, dron öğrenilen
        elips rotada hedefi TAKİP eder (yuva + carrot pure-pursuit + hız kanalı).
    --sim varsa: elips yörüngede uçan hedefi yakalama simülasyonu çalıştırılır.

Bu dosya İNCE bir harness'tir — asıl mantık modüllerden çekilir:
    navigation.telemetry_conditioner  → MeasurementGate, AltitudeLock
    navigation.orbit_lock             → OrbitLockTracker (rota öğren/kilitle/dondur)
    guidance.route_follow_guidance    → RouteFollowGuidance, GuidePointSlewLimiter
    guidance.pid_{yaw,pitch,throttle} → kanıtlanmış PID kontrolcüleri
    route_viz                         → koyu tema, canlı pano, elips sim hedefi
================================================================================
"""

import sys
import os
import time
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Proje kök dizinini Python yoluna ekle
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "python"))
sys.path.insert(0, os.path.dirname(__file__))

import yaml

from navigation.route_intercept_system import RoutePlanner, solve_intercept
from navigation.telemetry_conditioner import MeasurementGate, AltitudeLock
from navigation.orbit_lock import OrbitLockTracker
from guidance.route_follow_guidance import RouteFollowGuidance, GuidePointSlewLimiter
from guidance.pid_pitch import PitchController
from guidance.pid_throttle import ThrottleController
from guidance.pid_yaw import YawController
from route_viz import LiveDashboard, EllipseSimTarget

# Sabitler
# 50 Hz — config flight.dt ile aynı. 100 Hz'de PID türev terimleri telemetri
# örnekleme basamaklarını büyütüp yaw/pitch'te bang-bang üretiyordu.
DT_LOOP = 0.02
DRONE_SPEED_CMS = 2500.0  # Dron hızı: 25 m/s

# Simülasyon Ayarları
SIM_TARGET_SPEED = 1800.0  # cm/s (18 m/s)
SIM_CENTER_X = 20000.0     # cm (200 m)
SIM_CENTER_Y = 20000.0     # cm (200 m)
SIM_A = 15000.0            # cm (150 m)
SIM_B = 10000.0            # cm (100 m)
SIM_GPS_NOISE = 500.0      # cm
SIM_GPS_PERIOD = 1.0       # s — hedef GPS örnekleme periyodu (yarışma: max 1 Hz)


# ===========================================================================
# SİMÜLASYON MODU
# ===========================================================================
class InterceptSimulator:
    def __init__(self):
        self.planner = RoutePlanner(buffer_size=60, min_fit_points=15, residual_threshold=1500.0)
        self.target = EllipseSimTarget(SIM_CENTER_X, SIM_CENTER_Y, SIM_A, SIM_B,
                                       SIM_TARGET_SPEED, SIM_GPS_NOISE)

    def run(self):
        print("=" * 60)
        print("  Route Intercept — Yakalama Simülasyonu")
        print(f"  Hedef Hızı: {SIM_TARGET_SPEED/100:.1f} m/s | Dron Hızı: {DRONE_SPEED_CMS/100:.1f} m/s")
        print("=" * 60)

        dash = LiveDashboard(
            "Route Intercept System — Otonom Yakalama Simülasyonu",
            "Otonom Yakalama Haritası",
            target_label="Hedef GPS Noktaları")

        target_x_log, target_y_log = [], []
        drone_x_log, drone_y_log = [0.0], [0.0]  # Dron (0,0)'dan başlar
        t_hist, dist_log, res_log = [], [], []

        t = 0.0
        intercept_resolved = False
        step = 0
        # GPS örneklemesi 1 Hz'e sınırlanır (yarışma RATELIMIT koşulu). 50 Hz
        # beslemek buffer'ı ~1 s'lik arka indirger → elips fit dejenere olur.
        last_gps_t = -1e9

        while t < 80.0:
            tx_real, ty_real = self.target.truth(t)

            # GPS ölçümü yalnızca 1 Hz üretilir ve planner'a beslenir
            if t - last_gps_t >= SIM_GPS_PERIOD:
                tx_noisy, ty_noisy = self.target.measure(t)
                last_gps_t = t
                target_x_log.append(tx_noisy)
                target_y_log.append(ty_noisy)
                self.planner.add_observation(t, tx_noisy, ty_noisy)

            dx, dy = drone_x_log[-1], drone_y_log[-1]

            intercept_pt = None
            if self.planner.route.fitted:
                sol = solve_intercept(self.planner.route, t, (dx, dy), DRONE_SPEED_CMS)
                if sol is not None:
                    intercept_pt = sol['intercept_point']
                    ix, iy = intercept_pt
                    v_dx, v_dy = ix - dx, iy - dy
                    v_dist = math.hypot(v_dx, v_dy)
                    if v_dist > 100.0:  # Yakalanana kadar ilerle
                        step_dist = min(DRONE_SPEED_CMS * DT_LOOP, v_dist)
                        dx += (v_dx / v_dist) * step_dist
                        dy += (v_dy / v_dist) * step_dist
                    if v_dist < 200.0 and not intercept_resolved:
                        print(f"\n  [YAKALANDI!] Hedef t={t:.2f} saniyede elips üzerinde yakalandı.")
                        intercept_resolved = True
                dash.set_ellipse(self.planner.route)

            drone_x_log.append(dx)
            drone_y_log.append(dy)

            t_hist.append(t)
            dist_log.append(math.hypot(tx_real - dx, ty_real - dy))
            res_log.append(self.planner.last_residual if self.planner.last_residual is not None else 0.0)

            # Grafikleri 10 Hz frekansında güncelle (her 10 adımda bir)
            step += 1
            if step % 10 == 0:
                dash.update((target_x_log, target_y_log), (drone_x_log, drone_y_log),
                            tx_real, ty_real, dx, dy, t_hist, dist_log, res_log, intercept_pt)
                plt.pause(0.001)

            t += DT_LOOP

        print("Simülasyon tamamlandı.")
        dash.block()


# ===========================================================================
# CANLI SDK MODU
# ===========================================================================
class LiveInterceptRunner:
    def __init__(self, headless=False, duration=None, log_dir=None):
        self._running = True
        self.headless = headless          # True: GUI yok, CSV + rapor PNG üretilir
        self.duration = duration          # saniye; dolunca koşu otomatik biter
        self.log_dir = log_dir or os.getcwd()

        # --- Config + güdüm kontrolcüleri (config/pid_params.yaml) ---
        cfg_path = os.path.join(PROJECT_ROOT, "config", "pid_params.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.yaw_ctrl = YawController(self.config)
        self.pitch_ctrl = PitchController(self.config)
        self.throttle_ctrl = ThrottleController(self.config)
        self.guidance = RouteFollowGuidance(self.config)

    def _connect_sdk(self):
        import drone_sdk as drone
        print("  drone_sdk'ya bağlanılıyor (127.0.0.1:12345) ...")
        while self._running:
            if drone.connect():
                print("  ✓ drone_sdk Bağlantısı kuruldu!\n")
                return drone
            print("  ✗ Bağlantı yok — 2 saniye sonra tekrar denenecek ...")
            time.sleep(2.0)
        return None

    def run(self):
        drone = self._connect_sdk()
        if drone is None:
            return

        print("  Canlı Rota ve Yakalama Çözücü Başlatıldı. CTRL+C ile durdurun.\n")

        dash = None
        if not self.headless:
            dash = LiveDashboard(
                "Route Intercept System — Canlı Yakalama Analizi (drone_sdk)",
                "Canlı Yakalama Haritası")

        # --- Runner-seviyesi parametreler (kanallar modüllerde) ---
        ri_cfg = self.config.get("route_intercept", {})
        NOMINAL_TARGET_SPEED = float(ri_cfg.get("nominal_target_speed_cms", 1700.0))
        LOCK_STABLE_LAPS = int(ri_cfg.get("lock_stable_laps", 3))
        WAIT_FOR_LOCK = bool(ri_cfg.get("wait_for_lock", True))
        # Yakın alan: guide noktasının üstünden geçerken kerteriz açısı 180°
        # savrulur — 360° kendi etrafında dönüşlerin kök nedeni. Bu yarıçapta
        # dönüş komutu verilmez, burun sabit kalır.
        NEAR_FIELD_CM = 700.0

        # --- Telemetri koşullandırma + rota kilidi + guide yumuşatma ---
        gate = MeasurementGate(jump_reject_cm=50000.0, stale_timeout_s=2.5)
        alt_lock = AltitudeLock(buffer_size=30, jump_reject_cm=1500.0, reject_persist=8)
        tracker = OrbitLockTracker(
            buffer_size=60, min_fit_points=15, residual_threshold=1500.0,
            outlier_threshold_cm=5000.0, lock_stable_laps=LOCK_STABLE_LAPS,
            min_lap_points=15, period_coverage=0.0, verbose=True)
        guide_slew = GuidePointSlewLimiter(NOMINAL_TARGET_SPEED * 2.0)

        t_hist = []
        target_x_log, target_y_log = [], []
        drone_x_log, drone_y_log = [], []
        dist_log, res_log = [], []

        wait_msg_shown = False
        last_yaw_cmd = 0.0
        prev_heading = None            # ölçülen yaw hızı (rapor metriği) için
        log_rows = []

        t0 = time.time()
        prev_loop_t = t0
        step = 0

        try:
            while self._running:
                step_start = time.time()
                t_now = step_start - t0
                dt = max(0.005, min(0.1, step_start - prev_loop_t))
                prev_loop_t = step_start

                # Telemetriyi al
                t_data = drone.get_telemetry()
                d_pos = t_data["drone"]["position"]
                d_rot = t_data["drone"]["rotation"]
                d_vel = t_data["drone"]["velocity"]
                tar_pos = t_data["target"]["position"]

                dx, dy, dz = d_pos[0], d_pos[1], d_pos[2]
                tx_raw, ty_raw, tz = tar_pos[0], tar_pos[1], tar_pos[2]

                # Bağlantı başında telemetri henüz akmıyor olabilir
                if dx == 0.0 and dy == 0.0 and tx_raw == 0.0 and ty_raw == 0.0:
                    time.sleep(DT_LOOP)
                    continue

                # Ölçüm kapısı: değişim tespiti + sıçrama reddi + bayatlık
                g = gate.update(t_now, tx_raw, ty_raw)
                tx, ty = g.x, g.y
                meas_changed, data_valid, data_stale = g.meas_changed, g.data_valid, g.data_stale

                # İrtifa kilidi (sıçrama reddeden hareketli ortalama)
                locked_alt = alt_lock.update(tz, meas_changed, data_valid)
                if locked_alt is None:
                    time.sleep(DT_LOOP)
                    continue

                target_x_log.append(tx)
                target_y_log.append(ty)
                drone_x_log.append(dx)
                drone_y_log.append(dy)

                # Rota öğren/kilitle/dondur; aktif rotayı al
                active_route = tracker.step(t_now, tx, ty, meas_changed)

                intercept_pt = None
                guide_x, guide_y = tx, ty      # Varsayılan: ham hedefe git
                ring_v_des = None
                if active_route.fitted:
                    guide_x, guide_y, ring_v_des = self.guidance.guide(
                        active_route, dx, dy, tx, ty)
                    intercept_pt = (float(guide_x), float(guide_y))
                    if dash is not None:
                        dash.set_ellipse(active_route)

                t_hist.append(t_now)
                dist_log.append(math.hypot(tx - dx, ty - dy))
                res_log.append(tracker.last_residual if tracker.last_residual is not None else 0.0)

                # Guide noktası eğim-sınırı (basamak girişini engeller)
                guide_x, guide_y = guide_slew.update(guide_x, guide_y, dt)

                ldx = guide_x - dx
                ldy = guide_y - dy
                ldz = locked_alt - dz          # Ortalama yüksekliğe kilitli
                dist_2d = math.hypot(ldx, ldy)
                dist_3d = math.hypot(dist_2d, ldz)

                # Ölçülen yaw hızı (deg/s) — yalnızca rapor metriği
                heading = d_rot[2]
                if prev_heading is None:
                    yaw_rate_meas = 0.0
                else:
                    dh = heading - prev_heading
                    while dh > 180:
                        dh -= 360
                    while dh < -180:
                        dh += 360
                    yaw_rate_meas = dh / dt
                prev_heading = heading

                near_field = dist_2d < NEAR_FIELD_CM

                # İlk yörünge kilidine kadar BEKLE (hover): rota bilinmeden ham
                # hedefi kovalamak verimsiz ve savruk.
                waiting = WAIT_FOR_LOCK and not tracker.orbit_locked
                if waiting and not wait_msg_shown:
                    print("  ⏸ İlk yörünge kilidi bekleniyor — hover'da bekleniyor...")
                    wait_msg_shown = True
                elif not waiting and wait_msg_shown:
                    print("  ▶ Yörünge kilitli — takibe geçiliyor.")
                    wait_msg_shown = False

                # ---- YAW: kerteriz hatası → dönüş komutu ----
                yaw_error, yaw_cmd = self.yaw_ctrl.calculate(ldx, ldy, d_rot[2], dt)

                if waiting or data_stale or near_field:
                    # Bekleme / yeni ölçüm yok / guide üstündeyiz: dönme
                    yaw_cmd = 0.0
                    self.yaw_ctrl.inner_pid.integral = 0.0
                    self.yaw_ctrl.inner_pid.first_run = True
                    last_yaw_cmd = 0.0
                    pitch_align_err = 0.0   # düz uçuşta pitch tam yetkili
                elif not data_valid:
                    yaw_cmd = last_yaw_cmd
                    pitch_align_err = yaw_error
                else:
                    last_yaw_cmd = yaw_cmd
                    pitch_align_err = yaw_error

                # ---- İLERİ HIZ PROFİLİ ----
                v_des = self.guidance.desired_speed(ring_v_des, dist_2d, waiting, data_stale)

                # ---- PITCH: hız PID → istenen açı → angle-mode stick ----
                # Hız geri beslemesi İMZALI ileri hızdır (hız vektörünün burun
                # yönüne izdüşümü); skaler |speed| geri kayışı gizler.
                hd_rad = math.radians(d_rot[2])
                v_fwd = d_vel[0] * math.cos(hd_rad) + d_vel[1] * math.sin(hd_rad)
                target_elevation = math.degrees(math.atan2(ldz, max(dist_2d, 1.0)))
                smooth_pitch, pitch_cmd = self.pitch_ctrl.calculate(
                    state="TAKIP",
                    target_speed_cmd=v_des,
                    drone_speed=v_fwd,
                    target_elevation=target_elevation,
                    drone_pitch=d_rot[1],
                    yaw_error=pitch_align_err,
                    dt=dt,
                )

                # ---- THROTTLE: kilitli ortalama irtifaya PID (0 = hover) ----
                thr_cmd = self.throttle_ctrl.calculate(ldz, dt)

                fy = max(-1.0, min(1.0, yaw_cmd))
                fp = -max(-1.0, min(1.0, pitch_cmd))   # sim_bridge işaret düzeni
                f_thr = max(-1.0, min(1.0, thr_cmd))

                drone.set_control_surfaces(f_thr, fp, 0.0, fy, True)

                if self.headless:
                    # (Debug) Gerçek hedef mesafesi — SADECE rapor metriği
                    truth = drone.get_debug_truth()
                    if truth.get("available"):
                        tp = truth["target"]["position"]
                        true_dist = math.hypot(tp[0] - dx, tp[1] - dy)
                    else:
                        true_dist = -1.0
                    log_rows.append((
                        t_now, dx, dy, dz, tx, ty, tz, locked_alt,
                        dist_2d, dist_3d, yaw_error, d_rot[2], yaw_rate_meas,
                        v_fwd, v_des, fy, fp, f_thr, 0.0,
                        int(meas_changed), int(data_stale), int(near_field),
                        int(tracker.orbit_locked), true_dist,
                    ))

                if self.duration is not None and t_now >= self.duration:
                    print(f"\n  Süre doldu ({self.duration:.0f} s) — koşu bitiriliyor.")
                    break

                # Arayüzü güncelle (headless modda çizim yok)
                step += 1
                if dash is not None and step % 5 == 0 and target_x_log:
                    dash.update((target_x_log, target_y_log), (drone_x_log, drone_y_log),
                                tx, ty, dx, dy, t_hist, dist_log, res_log, intercept_pt)

                elapsed = time.time() - step_start
                sleep_t = DT_LOOP - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            print("\n  Canlı takip sonlandırıldı.")
        finally:
            try:
                drone.set_control_surfaces(0.0, 0.0, 0.0, 0.0, True)
            except Exception:
                pass
            drone.disconnect()
            if self.headless:
                self._write_report(log_rows)
            elif dash is not None:
                dash.block()

    def _write_report(self, log_rows):
        """Headless koşu verisini CSV'ye yazar, özet metrikleri basar ve
        özet grafiği PNG olarak kaydeder."""
        if not log_rows:
            print("  (rapor) Kayıtlı veri yok.")
            return

        cols = ["t", "dx", "dy", "dz", "tx", "ty", "tz", "locked_alt",
                "dist_2d", "dist_3d", "yaw_err", "heading", "yaw_rate",
                "d_speed", "v_des", "cmd_yaw", "cmd_pitch", "cmd_thr", "cmd_roll",
                "meas_changed", "data_stale", "near_field", "orbit_locked",
                "true_dist"]
        arr = np.array(log_rows, dtype=float)
        os.makedirs(self.log_dir, exist_ok=True)
        csv_path = os.path.join(self.log_dir, "route_intercept_run.csv")
        np.savetxt(csv_path, arr, delimiter=",", header=",".join(cols), comments="")

        t = arr[:, 0]
        c = {name: arr[:, i] for i, name in enumerate(cols)}

        # --- Özet metrikler ---
        dur = t[-1] - t[0]
        dh = np.diff(c["heading"])
        dh = (dh + 180.0) % 360.0 - 180.0
        revs = float(np.sum(np.abs(dh)) / 360.0)
        revs_net = float(abs(np.sum(dh)) / 360.0)
        yaw_rate_rms = float(np.sqrt(np.mean(c["yaw_rate"] ** 2)))
        spd = c["d_speed"]
        alt_err = c["locked_alt"] - c["dz"]
        lock_mask = c["orbit_locked"] > 0.5
        seg = lock_mask if lock_mask.any() else np.ones_like(t, dtype=bool)

        print("\n  " + "=" * 56)
        print("  KOŞU RAPORU")
        print(f"    Süre                : {dur:8.1f} s   ({len(t)} örnek)")
        print(f"    Toplam gövde dönüşü : {revs:8.2f} tur (birikimli |Δyaw|/360)")
        print(f"    Net dönüş (rota)    : {revs_net:8.2f} tur — fazlalık {revs - revs_net:.2f} tur savrulma")
        print(f"    Yaw hızı RMS        : {yaw_rate_rms:8.1f} deg/s")
        print(f"    Drone hızı          : ort {np.mean(spd):6.0f} / std {np.std(spd):5.0f} cm/s")
        print(f"    İrtifa hatası RMS   : {np.sqrt(np.mean(alt_err**2)):8.0f} cm")
        print(f"    Guide mesafesi (kilit sonrası): ort {np.mean(c['dist_2d'][seg]):6.0f}"
              f" / min {np.min(c['dist_2d'][seg]):6.0f} cm")
        tgt_dist = np.hypot(c["tx"] - c["dx"], c["ty"] - c["dy"])
        print(f"    HEDEF mesafesi (kilit sonrası): ort {np.mean(tgt_dist[seg]):6.0f}"
              f" / min {np.min(tgt_dist[seg]):6.0f} cm")
        truth_ok = c["true_dist"] >= 0.0
        if (truth_ok & seg).any():
            m = truth_ok & seg
            print(f"    GERÇEK mesafe (debug, kilit sonrası): ort {np.mean(c['true_dist'][m]):6.0f}"
                  f" / min {np.min(c['true_dist'][m]):6.0f} cm")
        print(f"    Bayat veri oranı    : {100.0*np.mean(c['data_stale']):8.1f} %")
        print(f"    Yakın alan oranı    : {100.0*np.mean(c['near_field']):8.1f} %")
        print(f"    CSV                 : {csv_path}")

        # --- Özet grafik ---
        plt.switch_backend("Agg")
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        fig.suptitle("Route Intercept — Headless Koşu Raporu", fontweight="bold")

        ax = axes[0, 0]
        ax.plot(c["tx"], c["ty"], ".", ms=2, color="tab:red", alpha=0.4, label="Hedef GPS")
        ax.plot(c["dx"], c["dy"], color="tab:blue", lw=1.2, label="Drone")
        ax.set_title("XY Yörünge"); ax.set_xlabel("X (cm)"); ax.set_ylabel("Y (cm)")
        ax.axis("equal"); ax.legend(); ax.grid(True, alpha=0.4)

        ax = axes[0, 1]
        ax.plot(t, c["dz"], color="tab:blue", label="Drone Z")
        ax.plot(t, c["tz"], ".", ms=2, color="tab:red", alpha=0.4, label="Hedef Z (ham)")
        ax.plot(t, c["locked_alt"], color="tab:green", label="Kilitli ort. irtifa")
        ax.set_title("İrtifa"); ax.set_xlabel("t (s)"); ax.set_ylabel("Z (cm)")
        ax.legend(); ax.grid(True, alpha=0.4)

        ax = axes[1, 0]
        ax.plot(t, c["dist_2d"], color="tab:orange", label="2B mesafe (guide)")
        ax2 = ax.twinx()
        ax2.plot(t, c["d_speed"], color="tab:cyan", lw=0.8, label="Drone hızı")
        ax2.plot(t, c["v_des"], color="tab:green", lw=0.8, ls="--", label="İstenen hız")
        ax.set_title("Mesafe ve Hız"); ax.set_xlabel("t (s)")
        ax.set_ylabel("Mesafe (cm)"); ax2.set_ylabel("Hız (cm/s)")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")
        ax.grid(True, alpha=0.4)

        ax = axes[1, 1]
        ax.plot(t, c["yaw_err"], color="tab:red", lw=0.8, label="Yaw hatası (°)")
        ax2 = ax.twinx()
        ax2.plot(t, c["cmd_yaw"], color="tab:purple", lw=0.8, label="Yaw cmd")
        ax2.plot(t, c["cmd_pitch"], color="tab:pink", lw=0.8, label="Pitch cmd")
        ax2.plot(t, c["cmd_thr"], color="tab:cyan", lw=0.8, label="Thr cmd")
        ax2.set_ylim(-1.1, 1.1)
        ax.set_title("Yaw Hatası ve Komutlar"); ax.set_xlabel("t (s)")
        ax.set_ylabel("Yaw hatası (°)"); ax2.set_ylabel("Komut [-1..1]")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")
        ax.grid(True, alpha=0.4)

        fig.tight_layout()
        png_path = os.path.join(self.log_dir, "route_intercept_run.png")
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        print(f"    Rapor PNG           : {png_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Route Intercept System Testi")
    parser.add_argument("--sim", action="store_true", help="Yakalama simülasyonu çalıştırır.")
    parser.add_argument("--headless", action="store_true",
                        help="Canlı modda GUI açmadan koşar; CSV + rapor PNG üretir.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Canlı koşu süresi (s); dolunca otomatik durur.")
    parser.add_argument("--log-dir", default=None,
                        help="Headless rapor/CSV çıktı dizini (varsayılan: çalışma dizini).")
    args = parser.parse_args()

    if args.sim:
        np.random.seed(42)
        sim = InterceptSimulator()
        sim.run()
    else:
        runner = LiveInterceptRunner(headless=args.headless,
                                     duration=args.duration,
                                     log_dir=args.log_dir)
        runner.run()


if __name__ == "__main__":
    main()
