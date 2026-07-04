"""
================================================================================
TEST: Fixed-Wing CTRV UKF Tracker — Canlı SDK Desteği & Simülasyon
================================================================================
Dosya: tests/python/test_fixed_wing_tracker.py

Çalıştırma:
    python tests/python/test_fixed_wing_tracker.py [--sim]

Mod seçimi:
    --sim bayrağı yoksa: drone_sdk üzerinden canlı hedef telemetrisi okunur.
    --sim bayrağı varsa: 3D loiter uçuşu yapan bir hedef uçak simüle edilir.
================================================================================
"""

import sys
import os
import time
import math
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse

# Proje kök dizinini Python yoluna ekle
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "python"))

from navigation.fixed_wing_tracker import FixedWingTracker

# Sabitler
DT_PREDICT = 0.1  # Tahmin adımı (10 Hz)
GPS_PERIOD = 1.0  # GPS ölçüm periyodu (1 Hz)

# Simülasyon Sabitleri
SIM_SPEED = 1800.0      # cm/s (18 m/s)
SIM_RADIUS = 25000.0    # cm (250 m)
SIM_GPS_NOISE_XY = 1500.0  # cm
SIM_GPS_NOISE_Z = 400.0    # cm
SIM_DURATION = 60.0


# ===========================================================================
# SİMÜLASYON MODU
# ===========================================================================
class TrackerSimulator:
    def __init__(self):
        self.tracker = FixedWingTracker()
        self.omega = SIM_SPEED / SIM_RADIUS

    def run(self):
        print("=" * 60)
        print("  Fixed-Wing UKF Tracker — Simülasyon Modu")
        print(f"  Hız: {SIM_SPEED/100:.1f} m/s | Yarıçap: {SIM_RADIUS/100:.1f} m")
        print("=" * 60)

        # Matplotlib Premium Koyu Tema Ayarları
        plt.rcParams.update({
            "figure.facecolor":  "#0f1117",
            "axes.facecolor":    "#1a1d27",
            "axes.edgecolor":    "#3a3d4d",
            "axes.labelcolor":   "#c8ccd8",
            "xtick.color":       "#7a7d8d",
            "ytick.color":       "#7a7d8d",
            "text.color":        "#c8ccd8",
            "grid.color":        "#2a2d3d",
            "grid.linestyle":    "--",
            "grid.alpha":        0.5,
            "font.family":       "DejaVu Sans",
        })

        plt.ion()
        fig = plt.figure(figsize=(15, 8.5))
        fig.suptitle(
            "Fixed-Wing CTRV UKF Tracker — Simülasyon Analizi",
            fontsize=14, fontweight="bold", color="#e8ecf8", y=0.96
        )

        gs = gridspec.GridSpec(2, 2, figure=fig, left=0.07, right=0.96,
                               top=0.88, bottom=0.08, hspace=0.32, wspace=0.22)

        ax_traj = fig.add_subplot(gs[:, 0])
        ax_alt = fig.add_subplot(gs[0, 1])
        ax_speed = fig.add_subplot(gs[1, 1])

        # Plot çizgileri
        line_real_2d, = ax_traj.plot([], [], color="#29b6f6", lw=2.0, label="Gerçek Yol", zorder=3)
        line_noisy_2d, = ax_traj.plot([], [], 'o', color="#ff5252", ms=4, alpha=0.6, label="GPS Ölçümü", zorder=2)
        line_ukf_2d, = ax_traj.plot([], [], color="#00e676", lw=2.5, label="UKF Filtrelenmiş Yol", zorder=4)
        ax_traj.set_xlabel("X (cm)")
        ax_traj.set_ylabel("Y (cm)")
        ax_traj.set_title("2D Yatay Konum Takibi & Belirsizlik Elipsleri", fontsize=11, fontweight="bold")
        ax_traj.legend(loc="upper left", facecolor="#1a1d27", edgecolor="#3a3d4d")
        ax_traj.grid(True)

        line_real_alt, = ax_alt.plot([], [], color="#29b6f6", lw=1.5, label="Gerçek Z")
        line_noisy_alt, = ax_alt.plot([], [], 'o', color="#ff5252", ms=3, alpha=0.5, label="GPS Z")
        line_ukf_alt, = ax_alt.plot([], [], color="#00e676", lw=2.0, label="UKF Z Tahmini")
        ax_alt.set_xlabel("Zaman (s)")
        ax_alt.set_ylabel("Yükseklik (cm)")
        ax_alt.set_title("Yükseklik (Z Ekseni) Profili", fontsize=10, fontweight="bold")
        ax_alt.legend(loc="upper right", facecolor="#1a1d27", edgecolor="#3a3d4d")
        ax_alt.grid(True)

        line_real_speed, = ax_speed.plot([], [], color="#29b6f6", lw=1.5, label="Gerçek Hız")
        line_ukf_speed, = ax_speed.plot([], [], color="#00e676", lw=2.0, label="UKF Hız Tahmini")
        ax_speed.set_xlabel("Zaman (s)")
        ax_speed.set_ylabel("Hız (cm/s)")
        ax_speed.set_title("Hız Tahmin Performansı", fontsize=10, fontweight="bold")
        ax_speed.legend(loc="upper right", facecolor="#1a1d27", edgecolor="#3a3d4d")
        ax_speed.grid(True)

        def get_covariance_ellipse(mean, cov, n_std=2.0, **kwargs):
            cov_pos = cov[:2, :2]
            vals, vecs = np.linalg.eigh(cov_pos)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            width = 2 * n_std * np.sqrt(max(0, vals[0]))
            height = 2 * n_std * np.sqrt(max(0, vals[1]))
            return Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)

        t_hist, real_x, real_y, real_z = [], [], [], []
        noisy_x, noisy_y, noisy_z = [], [], []
        ukf_x, ukf_y, ukf_z = [], [], []
        real_speeds, ukf_speeds = [], []
        ellipse_patches = []

        steps = int(SIM_DURATION / DT_PREDICT)
        t = 0.0
        gps_counter = 0.0

        for step in range(steps):
            # Gerçek yörünge (3D Helis/Loiter)
            rx = SIM_RADIUS * math.cos(self.omega * t)
            ry = SIM_RADIUS * math.sin(self.omega * t)
            rz = 8000.0 + 1000.0 * math.sin(0.1 * t)  # hafif dalgalı yükseklik

            # 1 Hz'de GPS ölçümü al
            gps_counter += DT_PREDICT
            if gps_counter >= GPS_PERIOD - 1e-9 or not self.tracker._initialized:
                gps_counter = 0.0
                nx = rx + np.random.normal(0, SIM_GPS_NOISE_XY)
                ny = ry + np.random.normal(0, SIM_GPS_NOISE_XY)
                nz = rz + np.random.normal(0, SIM_GPS_NOISE_Z)
                res = self.tracker.update(t, nx, ny, nz)
                
                noisy_x.append(nx)
                noisy_y.append(ny)
                noisy_z.append(nz)
            else:
                # Ölçüm gelmeyen anlarda UKF Tahmin adımı
                res = self.tracker.predict_only(t)

            px, py, pz = res['position']

            # Kayıtları güncelle
            t_hist.append(t)
            real_x.append(rx)
            real_y.append(ry)
            real_z.append(rz)
            ukf_x.append(px)
            ukf_y.append(py)
            ukf_z.append(pz)
            real_speeds.append(SIM_SPEED)
            ukf_speeds.append(res['speed'])

            # Grafikleri güncelle
            line_real_2d.set_data(real_x, real_y)
            line_noisy_2d.set_data(noisy_x, noisy_y)
            line_ukf_2d.set_data(ukf_x, ukf_y)

            # Belirsizlik elipsleri
            for patch in ellipse_patches:
                patch.remove()
            ellipse_patches.clear()

            P_scaled = self.tracker._ukf.P[:2, :2] * (self.tracker.SCALE ** 2)
            ellipse1 = get_covariance_ellipse((px, py), P_scaled, n_std=1.0, edgecolor="#00e676", facecolor="none", ls="-", lw=1.0, alpha=0.8)
            ellipse2 = get_covariance_ellipse((px, py), P_scaled, n_std=2.0, edgecolor="#00e676", facecolor="#00e676", ls="--", lw=0.6, alpha=0.15)
            ax_traj.add_patch(ellipse1)
            ax_traj.add_patch(ellipse2)
            ellipse_patches.extend([ellipse1, ellipse2])

            ax_traj.relim(); ax_traj.autoscale_view()
            ax_traj.set_aspect('equal', 'datalim')

            line_real_alt.set_data(t_hist, real_z)
            line_noisy_alt.set_data(t_hist[-len(noisy_z):], noisy_z)
            line_ukf_alt.set_data(t_hist, ukf_z)
            ax_alt.relim(); ax_alt.autoscale_view()

            line_real_speed.set_data(t_hist, real_speeds)
            line_ukf_speed.set_data(t_hist, ukf_speeds)
            ax_speed.relim(); ax_speed.autoscale_view()

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.02)
            t += DT_PREDICT

        plt.ioff()
        print("Simülasyon tamamlandı.")
        plt.show()


# ===========================================================================
# CANLI SDK MODU
# ===========================================================================
class LiveTrackerRunner:
    def __init__(self):
        self.tracker = FixedWingTracker()
        self._running = True

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

        print("  UKF Canlı Takip Başlatıldı. CTRL+C ile durdurun.\n")

        # Matplotlib Premium Koyu Tema Ayarları
        plt.rcParams.update({
            "figure.facecolor":  "#0f1117",
            "axes.facecolor":    "#1a1d27",
            "axes.edgecolor":    "#3a3d4d",
            "axes.labelcolor":   "#c8ccd8",
            "xtick.color":       "#7a7d8d",
            "ytick.color":       "#7a7d8d",
            "text.color":        "#c8ccd8",
            "grid.color":        "#2a2d3d",
            "grid.linestyle":    "--",
            "grid.alpha":        0.5,
            "font.family":       "DejaVu Sans",
        })

        plt.ion()
        fig = plt.figure(figsize=(15, 8.5))
        fig.suptitle(
            "Fixed-Wing CTRV UKF Tracker — Canlı drone_sdk Telemetri İzleme",
            fontsize=14, fontweight="bold", color="#e8ecf8", y=0.96
        )

        gs = gridspec.GridSpec(2, 2, figure=fig, left=0.07, right=0.96,
                               top=0.88, bottom=0.08, hspace=0.32, wspace=0.22)

        ax_traj = fig.add_subplot(gs[:, 0])
        ax_alt = fig.add_subplot(gs[0, 1])
        ax_speed = fig.add_subplot(gs[1, 1])

        # Plot çizgileri
        line_real_2d, = ax_traj.plot([], [], color="#29b6f6", lw=2.0, label="Gerçek Yol (Truth)", zorder=3)
        line_noisy_2d, = ax_traj.plot([], [], 'o', color="#ff5252", ms=4, alpha=0.6, label="Gürültülü GPS (Noisy)", zorder=2)
        line_ukf_2d, = ax_traj.plot([], [], color="#00e676", lw=2.5, label="UKF Filtrelenmiş Yol", zorder=4)
        ax_traj.set_xlabel("X (cm)")
        ax_traj.set_ylabel("Y (cm)")
        ax_traj.set_title("2D Yatay Konum Takibi & Belirsizlik Elipsleri", fontsize=11, fontweight="bold")
        ax_traj.legend(loc="upper left", facecolor="#1a1d27", edgecolor="#3a3d4d")
        ax_traj.grid(True)

        line_real_alt, = ax_alt.plot([], [], color="#29b6f6", lw=1.5, label="Gerçek Z")
        line_noisy_alt, = ax_alt.plot([], [], 'o', color="#ff5252", ms=3, alpha=0.5, label="Noisy Z")
        line_ukf_alt, = ax_alt.plot([], [], color="#00e676", lw=2.0, label="UKF Z Tahmini")
        ax_alt.set_xlabel("Zaman (s)")
        ax_alt.set_ylabel("Yükseklik (cm)")
        ax_alt.set_title("Yükseklik (Z Ekseni) Profili", fontsize=10, fontweight="bold")
        ax_alt.legend(loc="upper right", facecolor="#1a1d27", edgecolor="#3a3d4d")
        ax_alt.grid(True)

        line_real_speed, = ax_speed.plot([], [], color="#29b6f6", lw=1.5, label="Gerçek Hız")
        line_ukf_speed, = ax_speed.plot([], [], color="#00e676", lw=2.0, label="UKF Hız Tahmini")
        ax_speed.set_xlabel("Zaman (s)")
        ax_speed.set_ylabel("Hız (cm/s)")
        ax_speed.set_title("Hız Tahmin Performansı", fontsize=10, fontweight="bold")
        ax_speed.legend(loc="upper right", facecolor="#1a1d27", edgecolor="#3a3d4d")
        ax_speed.grid(True)

        def get_covariance_ellipse(mean, cov, n_std=2.0, **kwargs):
            cov_pos = cov[:2, :2]
            vals, vecs = np.linalg.eigh(cov_pos)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            width = 2 * n_std * np.sqrt(max(0, vals[0]))
            height = 2 * n_std * np.sqrt(max(0, vals[1]))
            return Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)

        t_hist, real_x, real_y, real_z = [], [], [], []
        noisy_x, noisy_y, noisy_z = [], [], []
        ukf_x, ukf_y, ukf_z = [], [], []
        real_speeds, ukf_speeds = [], []
        ellipse_patches = []

        # Kontrolcü Parametreleri (drone_gui.py ile birebir uyumlu)
        KP_YAW = 0.07
        KP_PITCH = 0.035
        BASE_HOVER_THR = 0.58
        max_allowed_thr = 1.0
        PITCH_SIGN = -1.0

        t0 = time.time()
        gps_counter = 0.0
        step = 0

        try:
            while self._running:
                step_start = time.time()
                t_now = step_start - t0

                # Telemetriyi oku
                t_data = drone.get_telemetry()
                d_pos = t_data["drone"]["position"]
                d_rot = t_data["drone"]["rotation"]
                d_speed = t_data["drone"]["speed"]
                tar_pos = t_data["target"]["position"]

                dx, dy, dz = d_pos[0], d_pos[1], d_pos[2]
                nx, ny, nz = tar_pos[0], tar_pos[1], tar_pos[2]

                # 1 Hz'de GPS ölçümü al
                gps_counter += DT_PREDICT
                if gps_counter >= GPS_PERIOD - 1e-9 or not self.tracker._initialized:
                    gps_counter = 0.0
                    res = self.tracker.update(t_now, nx, ny, nz)
                    noisy_x.append(nx)
                    noisy_y.append(ny)
                    noisy_z.append(nz)
                    
                    # Debug/Ground truth verisi al (varsa)
                    try:
                        truth = drone.get_debug_truth()
                    except Exception:
                        truth = {}
                    
                    if truth.get("available"):
                        tx = truth["target"]["position"][0]
                        ty = truth["target"]["position"][1]
                        tz = truth["target"]["position"][2]
                        ts = truth["target"].get("speed", np.nan)
                    else:
                        tx, ty, tz, ts = np.nan, np.nan, np.nan, np.nan

                    real_x.append(tx)
                    real_y.append(ty)
                    real_z.append(tz)
                    real_speeds.append(ts)
                else:
                    # Tahmin adımı
                    res = self.tracker.predict_only(t_now)

                px, py, pz = res['position']
                t_speed = res['speed']

                t_hist.append(t_now)
                ukf_x.append(px)
                ukf_y.append(py)
                ukf_z.append(pz)
                ukf_speeds.append(t_speed)

                # ====================================================================
                # DRONE YÖNLENDİRME KONTROLCÜSÜ (drone_gui.py mantığı)
                # ====================================================================
                # UKF ile tahmin edilen hedef konumuna yönelme:
                ldx = px - dx
                ldy = py - dy
                ldz = pz - dz

                dist_2d = math.hypot(ldx, ldy)
                dist_3d = math.hypot(dist_2d, ldz)

                # Yaw (Sapma) yönelim kontrolü
                target_yaw = math.degrees(math.atan2(ldy, ldx))
                yaw_error = target_yaw - d_rot[2]
                while yaw_error > 180:
                    yaw_error -= 360
                while yaw_error < -180:
                    yaw_error += 360
                
                final_yaw = yaw_error * KP_YAW
                if not hasattr(self, "_yaw_smooth"):
                    self._yaw_smooth = 0.0
                self._yaw_smooth = 0.7 * self._yaw_smooth + 0.3 * final_yaw
                final_yaw = self._yaw_smooth

                # Z ekseni açı ve gaz (throttle) yönelim kontrolü
                CAMERA_TILT = 0.0
                target_pitch_world = math.degrees(math.atan2(ldz, dist_2d if dist_2d > 1e-6 else 1e-6))
                lock_pitch = target_pitch_world - CAMERA_TILT

                PHASE2_START = 4500.0
                DIVE_OFFSET = -25.0
                match_target_thr = (BASE_HOVER_THR * 0.92) + (t_speed / 3800.0)
                alt_correction = ldz * 0.0015
                match_target_thr += alt_correction

                if dist_3d > PHASE2_START:
                    target_pitch_drone = lock_pitch + DIVE_OFFSET
                    final_thr = min(match_target_thr + 0.24, max_allowed_thr)
                else:
                    local_scale = max(0.0, dist_3d / PHASE2_START)
                    flare_trigger_scale = min(1.0, dist_3d / 1000.0)
                    flare_effect = (1.0 - flare_trigger_scale) * 15.0
                    target_pitch_drone = lock_pitch + (DIVE_OFFSET * local_scale) + flare_effect
                    approach_buffer = 0.22 * local_scale
                    final_thr = match_target_thr + approach_buffer

                speed_gap = max(0.0, t_speed - d_speed)
                final_thr += min(0.42, speed_gap / 3200.0)
                final_thr = min(max_allowed_thr, final_thr)

                # Pitch (Yunuslama) kontrolü
                pitch_error = target_pitch_drone - d_rot[1]
                final_pitch = PITCH_SIGN * pitch_error * KP_PITCH

                # Roll (Yatış) kontrolü (dönüşe yaslanma ve yatayda kalma eğilimi)
                bank = max(-0.3, min(0.3, yaw_error * 0.012))
                level = -d_rot[0] * 0.15
                final_roll = bank + level
                if not hasattr(self, "_roll_smooth"):
                    self._roll_smooth = 0.0
                self._roll_smooth = 0.50 * self._roll_smooth + 0.25 * final_roll
                final_roll = self._roll_smooth

                fy = max(-1.0, min(1.0, final_yaw))
                fp = max(-1.0, min(1.0, final_pitch))
                fr = max(-1.0, min(1.0, final_roll))
                f_thr = max(-0.8, min(max_allowed_thr, final_thr))

                # Komutları doğrudan API üzerinden drone'a gönder
                drone.set_yaw(fy)
                drone.set_pitch(fp)
                drone.set_throttle(f_thr)
                drone.set_roll(-fr)
                drone.set_arm(True)  # Otomatik ARM et

                # Arayüzü 2 Hz hızda güncelle (her 5 adımda bir)
                step += 1
                if step % 5 == 0 and noisy_x:
                    line_real_2d.set_data(real_x, real_y)
                    line_noisy_2d.set_data(noisy_x, noisy_y)
                    line_ukf_2d.set_data(ukf_x, ukf_y)

                    for patch in ellipse_patches:
                        patch.remove()
                    ellipse_patches.clear()

                    P_scaled = self.tracker._ukf.P[:2, :2] * (self.tracker.SCALE ** 2)
                    ellipse1 = get_covariance_ellipse((px, py), P_scaled, n_std=1.0, edgecolor="#00e676", facecolor="none", ls="-", lw=1.0, alpha=0.8)
                    ellipse2 = get_covariance_ellipse((px, py), P_scaled, n_std=2.0, edgecolor="#00e676", facecolor="#00e676", ls="--", lw=0.6, alpha=0.15)
                    ax_traj.add_patch(ellipse1)
                    ax_traj.add_patch(ellipse2)
                    ellipse_patches.extend([ellipse1, ellipse2])

                    ax_traj.relim(); ax_traj.autoscale_view()
                    ax_traj.set_aspect('equal', 'datalim')

                    line_real_alt.set_data(t_hist[-len(real_z):], real_z)
                    line_noisy_alt.set_data(t_hist[-len(noisy_z):], noisy_z)
                    line_ukf_alt.set_data(t_hist, ukf_z)
                    ax_alt.relim(); ax_alt.autoscale_view()

                    line_real_speed.set_data(t_hist[-len(real_speeds):], real_speeds)
                    line_ukf_speed.set_data(t_hist, ukf_speeds)
                    ax_speed.relim(); ax_speed.autoscale_view()

                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

                elapsed = time.time() - step_start
                sleep_t = DT_PREDICT - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            print("\n  Canlı takip sonlandırıldı.")
        finally:
            drone.disconnect()
            plt.ioff()
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Fixed-Wing UKF Tracker Testi")
    parser.add_argument("--sim", action="store_true", help="Loiter simülasyonu çalıştırır.")
    args = parser.parse_args()

    if args.sim:
        np.random.seed(42)
        sim = TrackerSimulator()
        sim.run()
    else:
        runner = LiveTrackerRunner()
        runner.run()


if __name__ == "__main__":
    main()
