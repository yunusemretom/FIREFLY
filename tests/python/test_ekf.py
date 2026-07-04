"""
================================================================================
TEST: Fixed-Wing CTRV EKF — Loiter Simülasyonu + Canlı SDK Desteği
================================================================================
Dosya: tests/python/test_ekf.py

Çalıştırma:
    cd /home/tom/Documents/Projeler/Firefly
    python tests/python/test_ekf.py [--sim]

Mod seçimi:
    --sim bayrağı yoksa: drone_sdk üzerinden canlı telemetri okunur.
                         Bağlantı gelene kadar beklenir (CTRL+C ile çık).
    --sim bayrağı varsa: 15 m/s hızda dairesel loiter simülasyonu çalışır.

Görselleştirme:
    - Sol panel  : 2D yörünge (gerçek / gürültülü GPS / EKF tahmini)
    - Sağ üst    : Konum hatası (cm) zaman serisi
    - Sağ orta   : EKF hız tahmini (cm/s)
    - Sağ alt    : EKF yaw-rate tahmini (°/s)
================================================================================
"""

import sys
import os
import time
import math
import argparse
import threading
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Proje kökünü Python yoluna ekle (src/python import için)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "python"))

from navigation.ekf import FixedWingEKF

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
DT_PREDICT   = 0.1        # saniye — EKF tahmin frekansı (10 Hz)
GPS_HZ       = 1.0        # GPS güncelleme frekansı (1 Hz)
GPS_PERIOD   = 1.0 / GPS_HZ

# Simülasyon parametreleri
SIM_SPEED_CMS   = 1500.0  # 15 m/s
SIM_RADIUS_CM   = 20000.0 # 200 m çaplı daire
SIM_GPS_NOISE   = 1500.0  # cm (±15 m GPS gürültüsü — yarışma koşulları)
SIM_DURATION_S  = 60.0    # simülasyon süresi (saniye)
SIM_SPIKE_PROB  = 0.05    # %5 ihtimalle GPS spike (ani zıplama)
SIM_SPIKE_MAG   = 8000.0  # cm — spike büyüklüğü


# ===========================================================================
# BÖLÜM 1 — SİMÜLASYON MODU (--sim bayrağı ile)
# ===========================================================================

class LoiterSimulator:
    """
    15 m/s hızda, 200 m yarıçaplı dairesel (loiter) yörünge simülatörü.
    1 Hz'de gürültülü GPS verisi üretir; EKF'i besler.
    """

    def __init__(self):
        self.ekf = FixedWingEKF()
        self.omega_true = SIM_SPEED_CMS / SIM_RADIUS_CM  # rad/s

        # Kayıt listeleri
        self.t_log        = []   # zaman
        self.true_x       = []   # gerçek x
        self.true_y       = []   # gerçek y
        self.gps_x        = []   # GPS x (gürültülü)
        self.gps_y        = []   # GPS y (gürültülü)
        self.ekf_x        = []   # EKF tahmini x
        self.ekf_y        = []   # EKF tahmini y
        self.ekf_v        = []   # EKF hız
        self.ekf_omega    = []   # EKF yaw rate
        self.pos_err      = []   # konum hatası (cm)
        self.t_gps_log    = []   # GPS zaman damgaları

    def _true_position(self, t: float):
        """Dairesel loiter'da gerçek konumu döner (cm)."""
        angle = self.omega_true * t
        x = SIM_RADIUS_CM * math.cos(angle)
        y = SIM_RADIUS_CM * math.sin(angle)
        return x, y

    def _noisy_gps(self, x_true: float, y_true: float, t: float):
        """Gerçekçi GPS ölçümü üretir (gürültü + spike)."""
        noise_x = np.random.normal(0, SIM_GPS_NOISE)
        noise_y = np.random.normal(0, SIM_GPS_NOISE)

        # %5 ihtimalle ani zıplama (spike)
        if np.random.random() < SIM_SPIKE_PROB:
            direction = np.random.uniform(0, 2 * math.pi)
            noise_x += SIM_SPIKE_MAG * math.cos(direction)
            noise_y += SIM_SPIKE_MAG * math.sin(direction)

        return x_true + noise_x, y_true + noise_y

    def run(self) -> dict:
        """Simülasyonu çalıştırır, kayıt verilerini döner."""
        print("=" * 60)
        print("  CTRV EKF — Loiter Simülasyonu")
        print(f"  Hız: {SIM_SPEED_CMS/100:.0f} m/s | Yarıçap: {SIM_RADIUS_CM/100:.0f} m")
        print(f"  GPS Gürültüsü: ±{SIM_GPS_NOISE/100:.0f} m | Spike: %{SIM_SPIKE_PROB*100:.0f}")
        print(f"  Süre: {SIM_DURATION_S:.0f} s | Predict: {1/DT_PREDICT:.0f} Hz | GPS: {GPS_HZ:.0f} Hz")
        print("=" * 60)

        steps       = int(SIM_DURATION_S / DT_PREDICT)
        gps_counter = 0.0
        t           = 0.0

        for step in range(steps):
            x_true, y_true = self._true_position(t)

            # --- EKF Predict (10 Hz) ---
            self.ekf.predict(DT_PREDICT)

            # --- GPS Update (1 Hz) ---
            gps_counter += DT_PREDICT
            if gps_counter >= GPS_PERIOD - 1e-9:
                gps_counter = 0.0
                x_gps, y_gps = self._noisy_gps(x_true, y_true, t)
                accepted = self.ekf.update(np.array([x_gps, y_gps]), timestamp=t)
                self.gps_x.append(x_gps)
                self.gps_y.append(y_gps)
                self.t_gps_log.append(t)

            # --- Kayıt ---
            self.t_log.append(t)
            self.true_x.append(x_true)
            self.true_y.append(y_true)

            ex, ey = self.ekf.position
            self.ekf_x.append(ex)
            self.ekf_y.append(ey)
            self.ekf_v.append(self.ekf.speed)
            self.ekf_omega.append(math.degrees(self.ekf.yaw_rate))

            if self.ekf.is_initialized:
                err = math.hypot(ex - x_true, ey - y_true)
            else:
                err = 0.0
            self.pos_err.append(err)

            t += DT_PREDICT

        print(f"\n  Ortalama Konum Hatası : {np.mean(self.pos_err[10:]):.1f} cm")
        print(f"  Maksimum Konum Hatası : {np.max(self.pos_err[10:]):.1f} cm")
        print(f"  Son EKF Hız Tahmini   : {self.ekf.speed/100:.2f} m/s")
        print(f"  Son EKF Yaw Rate      : {math.degrees(self.ekf.yaw_rate):.3f} °/s")

        return {
            "t":        np.array(self.t_log),
            "true_x":   np.array(self.true_x),
            "true_y":   np.array(self.true_y),
            "gps_x":    np.array(self.gps_x),
            "gps_y":    np.array(self.gps_y),
            "t_gps":    np.array(self.t_gps_log),
            "ekf_x":    np.array(self.ekf_x),
            "ekf_y":    np.array(self.ekf_y),
            "ekf_v":    np.array(self.ekf_v),
            "ekf_omega":np.array(self.ekf_omega),
            "pos_err":  np.array(self.pos_err),
        }


def plot_sim_results(data: dict):
    """Simülasyon sonuçlarını 4 panelde görselleştirir."""
    matplotlib.rcParams.update({
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

    fig = plt.figure(figsize=(16, 9), facecolor="#0f1117")
    fig.suptitle(
        "Fixed-Wing CTRV EKF — Loiter Simülasyonu  |  15 m/s  |  Ø 400 m",
        fontsize=15, fontweight="bold", color="#e8ecf8", y=0.97
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, left=0.06, right=0.97,
                           top=0.92, bottom=0.07, hspace=0.45, wspace=0.35)

    ax_traj = fig.add_subplot(gs[:, 0])  # Sol — yörünge (tüm satırlar)
    ax_err  = fig.add_subplot(gs[0, 1])  # Sağ üst — hata
    ax_v    = fig.add_subplot(gs[1, 1])  # Sağ orta — hız
    ax_om   = fig.add_subplot(gs[2, 1])  # Sağ alt — yaw rate

    t       = data["t"]
    t_gps   = data["t_gps"]

    # ---- Yörünge ----
    ax_traj.plot(data["true_x"] / 100, data["true_y"] / 100,
                 color="#4fc3f7", lw=1.5, label="Gerçek Yörünge", zorder=3)
    ax_traj.scatter(data["gps_x"] / 100, data["gps_y"] / 100,
                    c="#f44336", s=18, alpha=0.6, label="GPS Ölçümü (gürültülü)", zorder=2)
    ax_traj.plot(data["ekf_x"] / 100, data["ekf_y"] / 100,
                 color="#69f0ae", lw=2.0, label="EKF Tahmini", zorder=4)
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.set_title("2D Yörünge Karşılaştırması", fontsize=11, color="#e8ecf8")
    ax_traj.set_aspect("equal")
    ax_traj.grid(True)
    ax_traj.legend(loc="upper right", fontsize=8,
                   facecolor="#1a1d27", edgecolor="#3a3d4d")

    # ---- Konum Hatası ----
    ax_err.plot(t, data["pos_err"] / 100, color="#ffab40", lw=1.2)
    ax_err.axhline(np.mean(data["pos_err"][10:]) / 100, color="#ff7043",
                   ls="--", lw=1.0, label=f"Ort. {np.mean(data['pos_err'][10:])/100:.1f} m")
    ax_err.set_ylabel("Hata (m)")
    ax_err.set_title("EKF Konum Hatası", fontsize=10, color="#e8ecf8")
    ax_err.legend(fontsize=8, facecolor="#1a1d27", edgecolor="#3a3d4d")
    ax_err.grid(True)

    # ---- EKF Hız ----
    ax_v.plot(t, data["ekf_v"] / 100, color="#b39ddb", lw=1.2)
    ax_v.axhline(SIM_SPEED_CMS / 100, color="#ce93d8",
                 ls="--", lw=1.0, label=f"Gerçek {SIM_SPEED_CMS/100:.0f} m/s")
    ax_v.set_ylabel("Hız (m/s)")
    ax_v.set_title("EKF Hız Tahmini", fontsize=10, color="#e8ecf8")
    ax_v.legend(fontsize=8, facecolor="#1a1d27", edgecolor="#3a3d4d")
    ax_v.grid(True)

    # ---- EKF Yaw Rate ----
    omega_true_deg = math.degrees(SIM_SPEED_CMS / SIM_RADIUS_CM)
    ax_om.plot(t, data["ekf_omega"], color="#80cbc4", lw=1.2)
    ax_om.axhline(omega_true_deg, color="#4db6ac",
                  ls="--", lw=1.0, label=f"Gerçek {omega_true_deg:.2f} °/s")
    ax_om.set_xlabel("Zaman (s)")
    ax_om.set_ylabel("Yaw Rate (°/s)")
    ax_om.set_title("EKF Yaw-Rate Tahmini", fontsize=10, color="#e8ecf8")
    ax_om.legend(fontsize=8, facecolor="#1a1d27", edgecolor="#3a3d4d")
    ax_om.grid(True)

    plt.savefig(os.path.join(PROJECT_ROOT, "ekf_loiter_sim.png"),
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("\n  Grafik kaydedildi: ekf_loiter_sim.png")
    plt.show()


# ===========================================================================
# BÖLÜM 2 — CANLI SDK MODU (varsayılan)
# ===========================================================================

class LiveEKFRunner:
    """
    drone_sdk üzerinden hedef konumunu okuyarak EKF filtreler.
    Bağlantı gelene kadar bekler (CTRL+C ile sonlandırılabilir).
    Matplotlib real-time dashboard gösterir.
    """

    def __init__(self):
        self.ekf = FixedWingEKF()
        self._running = True

        # Veri tamponları (thread-safe değil ama sadece okunuyor GUI'de)
        self.t_log     = []
        self.raw_x     = []
        self.raw_y     = []
        self.ekf_x     = []
        self.ekf_y     = []
        self.ekf_v     = []
        self.ekf_omega = []
        self.pos_jump  = []   # raw GPS'in bir öncekinden farkı (cm)

    def _connect_sdk(self):
        """drone_sdk'ya bağlantı gelene kadar bekler."""
        # drone_sdk proje kökünden import edilmeli
        sdk_path = PROJECT_ROOT
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)

        import drone_sdk as drone

        print("  drone_sdk'ya bağlanılıyor (127.0.0.1:12345) ...")
        while self._running:
            ok = drone.connect()
            if ok:
                print("  ✓ Bağlantı kuruldu!\n")
                return drone
            print("  ✗ Bağlantı yok — 2 saniye sonra tekrar denenecek ...")
            time.sleep(2.0)
        return None

    def run(self):
        """Canlı EKF döngüsünü başlatır."""
        drone = self._connect_sdk()
        if drone is None:
            return

        print("  EKF canlı takip başlatıldı. CTRL+C ile dur.\n")

        # Matplotlib gerçek zamanlı grafik
        matplotlib.rcParams.update({
            "figure.facecolor": "#0f1117",
            "axes.facecolor":   "#1a1d27",
            "axes.edgecolor":   "#3a3d4d",
            "axes.labelcolor":  "#c8ccd8",
            "xtick.color":      "#7a7d8d",
            "ytick.color":      "#7a7d8d",
            "text.color":       "#c8ccd8",
            "grid.color":       "#2a2d3d",
            "grid.linestyle":   "--",
            "grid.alpha":       0.5,
        })

        plt.ion()
        fig = plt.figure(figsize=(14, 7), facecolor="#0f1117")
        fig.suptitle(
            "Fixed-Wing CTRV EKF — Canlı Hedef Takibi (drone_sdk)",
            fontsize=13, fontweight="bold", color="#e8ecf8"
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, left=0.08, right=0.97,
                               top=0.90, bottom=0.08, hspace=0.45, wspace=0.35)

        ax_traj = fig.add_subplot(gs[:, 0])
        ax_jump = fig.add_subplot(gs[0, 1])
        ax_v    = fig.add_subplot(gs[1, 1])

        for ax in [ax_traj, ax_jump, ax_v]:
            ax.grid(True)

        line_raw,  = ax_traj.plot([], [], "o", color="#f44336", ms=3,
                                  alpha=0.5, label="Ham GPS")
        line_ekf,  = ax_traj.plot([], [], "-", color="#69f0ae", lw=2,
                                  label="EKF Tahmini")
        ax_traj.legend(facecolor="#1a1d27", edgecolor="#3a3d4d", fontsize=8)
        ax_traj.set_title("2D Yörünge", fontsize=10, color="#e8ecf8")
        ax_traj.set_xlabel("X (cm)")
        ax_traj.set_ylabel("Y (cm)")

        line_jump, = ax_jump.plot([], [], color="#ffab40", lw=1.0)
        ax_jump.set_title("GPS Sıçrama Büyüklüğü (cm)", fontsize=10, color="#e8ecf8")
        ax_jump.set_xlabel("Zaman (s)")
        ax_jump.set_ylabel("Δ GPS (cm)")

        line_v, = ax_v.plot([], [], color="#b39ddb", lw=1.0)
        ax_v.set_title("EKF Hız Tahmini (cm/s)", fontsize=10, color="#e8ecf8")
        ax_v.set_xlabel("Zaman (s)")
        ax_v.set_ylabel("Hız (cm/s)")

        plt.show(block=False)

        # --- Ana döngü ---
        t0          = time.time()
        gps_counter = 0.0
        prev_raw    = None
        prev_gps_xy = None
        step        = 0

        try:
            while self._running:
                step_start = time.time()
                t_now = step_start - t0

                # EKF predict (her adım)
                self.ekf.predict(DT_PREDICT)

                # GPS güncelleme (1 Hz)
                gps_counter += DT_PREDICT
                if gps_counter >= GPS_PERIOD - 1e-9:
                    gps_counter = 0.0
                    pos = drone.get_target_location()   # (X, Y, Z) cm
                    raw_xy = np.array([pos[0], pos[1]])

                    # Sıçrama büyüklüğü hesapla
                    if prev_gps_xy is not None:
                        jump = float(np.linalg.norm(raw_xy - prev_gps_xy))
                    else:
                        jump = 0.0
                    prev_gps_xy = raw_xy.copy()

                    self.ekf.update(raw_xy, timestamp=t_now)
                    self.raw_x.append(raw_xy[0])
                    self.raw_y.append(raw_xy[1])
                    self.pos_jump.append(jump)
                    self.t_log.append(t_now)

                ex, ey = self.ekf.position
                self.ekf_x.append(ex)
                self.ekf_y.append(ey)
                self.ekf_v.append(self.ekf.speed)

                step += 1

                # Dashboard güncelleme (her 5 adımda bir = 2 Hz)
                if step % 5 == 0 and self.raw_x:
                    t_arr  = np.array(self.t_log)
                    raw_xa = np.array(self.raw_x)
                    raw_ya = np.array(self.raw_y)
                    ekf_xa = np.array(self.ekf_x[-len(self.raw_x):])
                    ekf_ya = np.array(self.ekf_y[-len(self.raw_y):])
                    jump_a = np.array(self.pos_jump)
                    v_a    = np.array(self.ekf_v[-len(self.t_log):])

                    line_raw.set_data(raw_xa, raw_ya)
                    line_ekf.set_data(ekf_xa, ekf_ya)
                    ax_traj.relim(); ax_traj.autoscale_view()

                    line_jump.set_data(t_arr, jump_a)
                    ax_jump.relim(); ax_jump.autoscale_view()

                    line_v.set_data(t_arr, v_a)
                    ax_v.relim(); ax_v.autoscale_view()

                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

                # Adım süresini DT_PREDICT'e sabitle
                elapsed = time.time() - step_start
                sleep_t = DT_PREDICT - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            print("\n  Durduruldu.")

        finally:
            drone.disconnect()
            plt.ioff()
            print("\n  Bağlantı kapatıldı.")
            if self.raw_x:
                plt.savefig(
                    os.path.join(PROJECT_ROOT, "ekf_live_session.png"),
                    dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
                )
                print("  Grafik kaydedildi: ekf_live_session.png")
            plt.show()


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fixed-Wing CTRV EKF Test — Simülasyon veya Canlı SDK"
    )
    parser.add_argument(
        "--sim", action="store_true",
        help="Canlı SDK yerine loiter simülasyonu çalıştır"
    )
    args = parser.parse_args()

    if args.sim:
        # ---- Simülasyon modu ----
        np.random.seed(42)
        sim  = LoiterSimulator()
        data = sim.run()
        plot_sim_results(data)
    else:
        # ---- Canlı SDK modu (varsayılan) ----
        runner = LiveEKFRunner()
        runner.run()


if __name__ == "__main__":
    main()
