"""
================================================================================
TEST: Route Kamikaze System — Canlı SDK & Simülasyon Taarruz (Vuruş) Testi
================================================================================
Dosya: tests/python/test_route_kamikaze.py

Çalıştırma:
    python tests/python/test_route_kamikaze.py [--sim] [--headless --duration N]

Fark (test_route_intercept.py'a göre):
    Bu test hedefi ARKASINDAN TAKİP ETMEZ. Hedefin tekrarlayan elips rotasını
    öğrenir, rota üzerinde en erken YAKALAMA (lead intercept) noktasını çözer
    ve tam taarruz hızıyla o noktaya DALARAK hedefi VURUR (kamikaze).

    - Takip yuvası / follow_distance YOK: guide noktası çarpışma noktasıdır.
    - Hız kanalı sabit taarruz hızında tutulur (yetişme değil, kapanma).
    - Eş-irtifa (kilitli ortalama hedef Z) + 2B kesişim → 3B çarpışma.
    - Son menzilde (terminal) çözücü bırakılır, doğrudan öngörülen hedefe dalar.
    - VURUŞ ONAYI = BAĞLANTI KOPMASI: taarruz başladıktan sonra SDK bağlantısı
      koparsa (dron hedefe çarpıp imha olmuş, sim soketi kapatmıştır) çarpışma
      onaylanmış sayılır. Mesafe yarıçapıyla erken bitirme YOK — bağlantı
      kopana kadar taarruza devam edilir.
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

# Proje kök dizinini Python yoluna ekle
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "python"))

import yaml

from navigation.route_intercept_system import RoutePlanner, EllipseRouteModel, solve_intercept
from guidance.pid_pitch import PitchController
from guidance.pid_throttle import ThrottleController
from guidance.pid_yaw import YawController

# Sabitler
# 50 Hz — config flight.dt ile aynı. 100 Hz'de PID türev terimleri telemetri
# örnekleme basamaklarını büyütüp yaw/pitch'te bang-bang üretiyordu.
DT_LOOP = 0.02
DRONE_SPEED_CMS = 2800.0  # Dron taarruz hızı: 28 m/s


def phase_intercept(route, phi_now, drone_pos, speed, horizon=20.0, ds=0.1):
    """Faz-DEMİRLİ çarpışma noktası çözücü.

    solve_intercept model saatini (position_at) kullanır; donmuş rotanın
    omega hatası uzun koşuda faz kaymasıyla nişanı ~10 m saptırır. Burada
    hedefin GERÇEK anlık fazı (phi_now, ölçümden demirlenmiş) baz alınıp
    gelecek konumlar phi_now + omega·s ile üretilir. Dronun 'speed' ile s
    saniyede kat edebildiği menzil = hedefe uzaklık olan ilk s* bulunur.
    Dönüş: çarpışma noktası (x, y) veya yakalanamıyorsa None.
    """
    dx, dy = drone_pos
    s = np.arange(0.0, horizon, ds)
    phis = phi_now + route.omega * s
    xt = route.a * np.cos(phis)
    yt = route.b * np.sin(phis)
    ex = route.cx + xt * np.cos(route.theta) - yt * np.sin(route.theta)
    ey = route.cy + xt * np.sin(route.theta) + yt * np.cos(route.theta)
    dist = np.hypot(ex - dx, ey - dy)
    f = speed * s - dist
    sign_change = np.where(np.diff(np.sign(f)) > 0)[0]
    if len(sign_change) == 0:
        return None
    i = sign_change[0]
    s1, s2, f1, f2 = s[i], s[i + 1], f[i], f[i + 1]
    s_star = s1 - f1 * (s2 - s1) / (f2 - f1)
    phi_star = phi_now + route.omega * s_star
    return route.point_on_ellipse(phi_star)

# Simülasyon Ayarları
SIM_TARGET_SPEED = 1800.0  # cm/s (18 m/s)
SIM_CENTER_X = 20000.0     # cm (200 m)
SIM_CENTER_Y = 20000.0     # cm (200 m)
SIM_A = 15000.0            # cm (150 m)
SIM_B = 10000.0            # cm (100 m)
SIM_GPS_NOISE = 500.0      # cm
SIM_HIT_RADIUS = 300.0     # cm — bu mesafeye girince VURUŞ


# ===========================================================================
# SİMÜLASYON MODU — Kamikaze Taarruz
# ===========================================================================
class KamikazeSimulator:
    def __init__(self):
        self.planner = RoutePlanner(buffer_size=60, min_fit_points=15, residual_threshold=1500.0)
        self.omega = SIM_TARGET_SPEED / math.sqrt((SIM_A**2 + SIM_B**2)/2.0)

    def run(self):
        print("=" * 60)
        print("  Route Kamikaze — Taarruz (Vuruş) Simülasyonu")
        print(f"  Hedef Hızı: {SIM_TARGET_SPEED/100:.1f} m/s | Dron Hızı: {DRONE_SPEED_CMS/100:.1f} m/s")
        print("=" * 60)

        # Matplotlib Premium Koyu Tema
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
            "Route Kamikaze System — Otonom Taarruz Simülasyonu",
            fontsize=14, fontweight="bold", color="#e8ecf8", y=0.96
        )

        gs = gridspec.GridSpec(2, 2, figure=fig, left=0.07, right=0.96,
                               top=0.88, bottom=0.08, hspace=0.32, wspace=0.22)

        ax_map = fig.add_subplot(gs[:, 0])
        ax_dist = fig.add_subplot(gs[0, 1])
        ax_res = fig.add_subplot(gs[1, 1])

        # Arayüz Elemanları
        line_target_path, = ax_map.plot([], [], 'o', color="#ff5252", ms=3, alpha=0.4, label="Hedef GPS Noktaları")
        line_fitted_ellipse, = ax_map.plot([], [], color="#ce93d8", lw=2, label="Tahmini Rota (Fitted Ellipse)")
        line_drone_path, = ax_map.plot([], [], color="#29b6f6", lw=2, label="Dron Rotası")

        point_target, = ax_map.plot([], [], 'o', color="#ff1744", ms=8, label="Hedef (Şu anki)")
        point_drone, = ax_map.plot([], [], '^', color="#00e5ff", ms=8, label="Dron")
        point_intercept, = ax_map.plot([], [], '*', color="#00e676", ms=12, label="Vuruş Noktası (Intercept)")
        line_intercept_vector, = ax_map.plot([], [], '--', color="#00e676", alpha=0.7)

        ax_map.set_xlabel("X (cm)")
        ax_map.set_ylabel("Y (cm)")
        ax_map.set_title("Otonom Taarruz Haritası", fontsize=11, fontweight="bold")
        ax_map.legend(loc="upper left", facecolor="#1a1d27", edgecolor="#3a3d4d")
        ax_map.grid(True)

        line_dist, = ax_dist.plot([], [], color="#ffab40", lw=1.5, label="Hedef-Dron Mesafesi")
        ax_dist.set_xlabel("Zaman (s)")
        ax_dist.set_ylabel("Mesafe (cm)")
        ax_dist.set_title("Dron ve Hedef Arasındaki Anlık Mesafe", fontsize=10, fontweight="bold")
        ax_dist.grid(True)

        line_res, = ax_res.plot([], [], color="#b39ddb", lw=1.5, label="Elips Fit Kalıntısı")
        ax_res.set_xlabel("Zaman (s)")
        ax_res.set_ylabel("Hata (cm)")
        ax_res.set_title("Elips Rota Kalıntı Hatası (Residuals)", fontsize=10, fontweight="bold")
        ax_res.grid(True)

        # Veri Kayıtları
        t_hist = []
        target_x_log, target_y_log = [], []
        drone_x_log, drone_y_log = [0.0], [0.0]  # Dron (0,0)'dan başlar
        dist_log, res_log = [], []

        t = 0.0
        hit_resolved = False
        step = 0

        while t < 80.0:
            step_start = time.time()

            # Hedef gerçek hareketi + GPS gürültüsü
            tx_real = SIM_CENTER_X + SIM_A * math.cos(self.omega * t)
            ty_real = SIM_CENTER_Y + SIM_B * math.sin(self.omega * t)
            tx_noisy = tx_real + np.random.normal(0, SIM_GPS_NOISE)
            ty_noisy = ty_real + np.random.normal(0, SIM_GPS_NOISE)

            target_x_log.append(tx_noisy)
            target_y_log.append(ty_noisy)

            # Route Planner'ı besle
            status = self.planner.add_observation(t, tx_noisy, ty_noisy)

            # Dronun anlık konumu
            dx, dy = drone_x_log[-1], drone_y_log[-1]

            # Eğer elips fit edilmişse vuruş noktasını çöz
            intercept_pt = None
            if self.planner.route.fitted and not hit_resolved:
                sol = solve_intercept(self.planner.route, t, (dx, dy), DRONE_SPEED_CMS)
                if sol is not None:
                    intercept_pt = sol['intercept_point']

                    # Dronu vuruş noktasına tam hızla yönlendir (kamikaze dalış)
                    ix, iy = intercept_pt
                    v_dx = ix - dx
                    v_dy = iy - dy
                    v_dist = math.hypot(v_dx, v_dy)

                    if v_dist > 1.0:
                        step_dist = min(DRONE_SPEED_CMS * DT_LOOP, v_dist)
                        dx += (v_dx / v_dist) * step_dist
                        dy += (v_dy / v_dist) * step_dist

                # Gerçek hedefe olan mesafeyle vuruş kontrolü
                if math.hypot(tx_real - dx, ty_real - dy) < SIM_HIT_RADIUS:
                    print(f"\n  [🎯 HEDEF VURULDU!] t={t:.2f} s — çarpışma yarıçapı "
                          f"{SIM_HIT_RADIUS/100:.1f} m içinde.")
                    hit_resolved = True

                # Elips çizgisi için noktaları üret
                phis = np.linspace(-np.pi, np.pi, 200)
                ex = self.planner.route.cx + self.planner.route.a * np.cos(phis) * np.cos(self.planner.route.theta) - self.planner.route.b * np.sin(phis) * np.sin(self.planner.route.theta)
                ey = self.planner.route.cy + self.planner.route.a * np.cos(phis) * np.sin(self.planner.route.theta) + self.planner.route.b * np.sin(phis) * np.cos(self.planner.route.theta)
                line_fitted_ellipse.set_data(ex, ey)

            drone_x_log.append(dx)
            drone_y_log.append(dy)

            t_hist.append(t)
            dist_log.append(math.hypot(tx_real - dx, ty_real - dy))
            res_log.append(self.planner.last_residual if self.planner.last_residual is not None else 0.0)

            # Grafikleri 10 Hz frekansında güncelle (her 10 adımda bir)
            step += 1
            if step % 10 == 0:
                line_target_path.set_data(target_x_log, target_y_log)
                line_drone_path.set_data(drone_x_log, drone_y_log)
                point_target.set_data([tx_real], [ty_real])
                point_drone.set_data([dx], [dy])

                if intercept_pt is not None:
                    point_intercept.set_data([intercept_pt[0]], [intercept_pt[1]])
                    line_intercept_vector.set_data([dx, intercept_pt[0]], [dy, intercept_pt[1]])
                else:
                    point_intercept.set_data([], [])
                    line_intercept_vector.set_data([], [])

                ax_map.relim(); ax_map.autoscale_view()
                ax_map.set_aspect('equal', 'datalim')

                line_dist.set_data(t_hist, dist_log)
                ax_dist.relim(); ax_dist.autoscale_view()

                line_res.set_data(t_hist, res_log)
                ax_res.relim(); ax_res.autoscale_view()

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)

            if hit_resolved:
                # Vuruş sonrası kısa bir süre grafiği açık tut
                break

            t += DT_LOOP

        plt.ioff()
        print("Simülasyon tamamlandı.")
        plt.show()


# ===========================================================================
# CANLI SDK MODU — Kamikaze Taarruz
# ===========================================================================
class LiveKamikazeRunner:
    def __init__(self, headless=False, duration=None, log_dir=None):
        self.planner = RoutePlanner(buffer_size=60, min_fit_points=15, residual_threshold=1500.0)
        self._running = True
        self.headless = headless          # True: GUI yok, CSV + rapor PNG üretilir
        self.duration = duration          # saniye; dolunca koşu otomatik biter
        self.log_dir = log_dir or os.getcwd()

        # --- Güdüm kontrolcüleri (config/pid_params.yaml) ---
        # sim_bridge.py ile birebir aynı, kanıtlanmış katman kullanılır:
        #  - YawController: kerteriz hatası → yaw rate stick (PID)
        #  - PitchController: hız PID → istenen açı → angle-mode stick
        #  - ThrottleController: irtifa PID (taban 0 = hover)
        cfg_path = os.path.join(PROJECT_ROOT, "config", "pid_params.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.yaw_ctrl = YawController(self.config)
        self.pitch_ctrl = PitchController(self.config)
        self.throttle_ctrl = ThrottleController(self.config)

        # KAMİKAZE irtifa yetkisi: hızlı ileri uçuş (yüksek burun eğimi) bu
        # simde tırmanma coupling'i üretiyor — throttle -0.5 komutta bile dron
        # tırmanıyordu ve integral yetkisi yok denecek kadar azdı (ki·int_limit
        # ≈ 0.04), ~2.2 m kalıcı yükseklik hatası bırakıyordu. Vuruş dikey
        # kaçtığı için throttle'a coupling'i yenecek ve steady-state'i sıfırlayacak
        # daha fazla yetki verilir (yalnızca bu runner'a özel; config bozulmaz).
        self.throttle_ctrl.alt_pid.kp = 0.40
        self.throttle_ctrl.alt_pid.ki = 0.06
        self.throttle_ctrl.alt_pid.integral_limit = 12.0

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

        print("  Canlı Kamikaze Taarruz Çözücü Başlatıldı. CTRL+C ile durdurun.\n")

        if not self.headless:
            # Matplotlib Premium Koyu Tema
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
                "Route Kamikaze System — Canlı Taarruz Analizi (drone_sdk)",
                fontsize=14, fontweight="bold", color="#e8ecf8", y=0.96
            )

            gs = gridspec.GridSpec(2, 2, figure=fig, left=0.07, right=0.96,
                                   top=0.88, bottom=0.08, hspace=0.32, wspace=0.22)

            ax_map = fig.add_subplot(gs[:, 0])
            ax_dist = fig.add_subplot(gs[0, 1])
            ax_res = fig.add_subplot(gs[1, 1])

            # Çizim Elemanları
            line_target_path, = ax_map.plot([], [], 'o', color="#ff5252", ms=3, alpha=0.4, label="Hedef GPS")
            line_fitted_ellipse, = ax_map.plot([], [], color="#ce93d8", lw=2, label="Tahmini Rota (Elips)")
            line_drone_path, = ax_map.plot([], [], color="#29b6f6", lw=1.5, label="Dron Gerçek Yolu")

            point_target, = ax_map.plot([], [], 'o', color="#ff1744", ms=8, label="Hedef (Şu anki)")
            point_drone, = ax_map.plot([], [], '^', color="#00e5ff", ms=8, label="Dron")
            point_intercept, = ax_map.plot([], [], '*', color="#00e676", ms=12, label="Vuruş Noktası (Intercept)")
            line_intercept_vector, = ax_map.plot([], [], '--', color="#00e676", alpha=0.7)

            ax_map.set_xlabel("X (cm)")
            ax_map.set_ylabel("Y (cm)")
            ax_map.set_title("Canlı Taarruz Haritası", fontsize=11, fontweight="bold")
            ax_map.legend(loc="upper left", facecolor="#1a1d27", edgecolor="#3a3d4d")
            ax_map.grid(True)

            line_dist, = ax_dist.plot([], [], color="#ffab40", lw=1.5, label="Hedef-Dron Mesafesi")
            ax_dist.set_xlabel("Zaman (s)")
            ax_dist.set_ylabel("Mesafe (cm)")
            ax_dist.set_title("Dron ve Hedef Arasındaki Mesafe", fontsize=10, fontweight="bold")
            ax_dist.grid(True)

            line_res, = ax_res.plot([], [], color="#b39ddb", lw=1.5, label="Elips Kalıntı Hatası")
            ax_res.set_xlabel("Zaman (s)")
            ax_res.set_ylabel("Hata (cm)")
            ax_res.set_title("Kalıntı Hatası (Residuals)", fontsize=10, fontweight="bold")
            ax_res.grid(True)

        t_hist = []
        target_x_log, target_y_log = [], []
        drone_x_log, drone_y_log = [], []
        dist_log, res_log = [], []

        # --- Kamikaze parametreleri (config: route_kamikaze) ---
        # Hedefin hız kanalı GÜVENİLMEZ; rota şeklinden çözülen vuruş noktasına
        # sabit taarruz hızıyla dalınır (yetişme/yuva mantığı YOK).
        rk_cfg = self.config.get("route_kamikaze", {})
        ri_cfg = self.config.get("route_intercept", {})
        KAMIKAZE_SPEED = float(rk_cfg.get("attack_speed_cms",
                               self.config["flight"].get("chase_speed", 2800.0)))
        TERMINAL_RANGE_CM = float(rk_cfg.get("terminal_range_cm", 3000.0))
        HIT_RADIUS_CM = float(rk_cfg.get("hit_radius_cm", 500.0))
        # Bağlantı kopması VURUŞ sayılır — AMA yalnızca dron hedefe yakınken.
        # Sim bazen çarpışma olmadan da donup soketi kapatıyor (dron 50 m
        # uzaktayken); bu menzilden uzaksa kopma "sim donması" olarak elenir.
        CONFIRM_RANGE_CM = float(rk_cfg.get("confirm_range_cm", 2000.0))
        # Commit (anti-spin): bu menzilin altında yaw yetkisi kısılır; dron
        # bearing'i kovalayıp spinlemek yerine düz dalıp hedefin içinden geçer.
        COMMIT_RANGE_CM = float(rk_cfg.get("commit_range_cm", 1500.0))
        YAW_COMMIT_CAP = float(rk_cfg.get("yaw_commit_cap", 0.20))
        # Son yaklaşmada hedef irtifasının bu kadar ALTINA nişan al → irtifadan
        # geçerek çarp (dron kalıcı ~2 m yukarıda kalıp ıskalamasın).
        TERMINAL_DESCENT_CM = float(rk_cfg.get("terminal_descent_cm", 400.0))
        LEAD_TIME_S = float(rk_cfg.get("lead_time_s", ri_cfg.get("lead_time_s", 1.0)))
        WAIT_FOR_LOCK = bool(rk_cfg.get("wait_for_lock", ri_cfg.get("wait_for_lock", True)))
        LOCK_STABLE_LAPS = int(rk_cfg.get("lock_stable_laps",
                               ri_cfg.get("lock_stable_laps", 2)))
        CHASE_MAX_CMS = self.config["pursuit"]["max_speed_cms"]
        MIN_SPEED_CMS = self.config["pursuit"]["min_speed_cms"]
        wait_msg_shown = False

        # Yörünge Kilitleme parametreleri
        OUTLIER_THRESHOLD_CM = 5000.0   # Elipsten bu mesafeden uzak noktalar outlier
        master_route = EllipseRouteModel()
        orbit_locked = False
        route_frozen = False          # yakınsama sonrası rotayı dondur
        orbit_buf_t, orbit_buf_x, orbit_buf_y = [], [], []
        phase_acc = 0.0
        last_target_phi = None
        last_guide = None             # guide noktası eğim-sınırı için önceki değer
        orbit_count = 0

        # Yükseklik sabitleme — hedefin gerçek irtifası değişmediği için Z,
        # ani sıçramaları reddeden bir hareketli ortalamayla tutulur. Eş-irtifa
        # kilidi 2B kesişimi 3B çarpışmaya çevirir.
        alt_buffer = []
        ALT_BUFFER_SIZE = 30
        ALT_JUMP_REJECT_CM = 1500.0
        ALT_REJECT_PERSIST = 8
        alt_reject_count = 0
        locked_alt = None

        # Ölçüm tazeliği — SDK yeni paket gelmese de son değeri tutar (stale
        # hold); konum DEĞİŞMEDİKÇE yeni ölçüm yok demektir.
        STALE_TIMEOUT_S = 2.5
        last_meas_t = None

        # Son komut hafızası
        last_yaw_cmd = 0.0
        last_tx, last_ty = None, None
        prev_heading = None
        log_rows = []

        # Faz izleyici (PLL): hedefin elips fazını GPS gürültüsüne karşı
        # yumuşatır — omega ile ilerletilir, ölçümle küçük katsayıyla düzeltilir.
        phi_track = None
        last_range = None             # son taze tahmini hedef menzili (vuruş teyidi)

        # Vuruş durumu — VURUŞ ONAYI bağlantı kopmasıyla verilir (aşağıda).
        impact = False
        attack_started = False        # kilit sonrası taarruza fiilen geçildi mi
        min_true_dist = float("inf")
        min_meas_dist = float("inf")
        impact_t = None

        t0 = time.time()
        prev_loop_t = t0
        step = 0

        try:
            while self._running:
                step_start = time.time()
                t_now = step_start - t0
                dt = max(0.005, min(0.1, step_start - prev_loop_t))
                prev_loop_t = step_start

                # BAĞLANTI KOPMASI = VURUŞ ONAYI. Arka plan alım döngüsü karşı
                # taraf soketi kapatınca is_connected'ı False yapar; geçici GPS
                # dropout'ları (socket.timeout) bağlantıyı KOPARMAZ. Taarruz
                # başladıysa bu kopma dronun hedefe çarpıp imha olması demektir.
                if not drone.is_connected():
                    close_enough = (last_range is not None
                                    and last_range < CONFIRM_RANGE_CM)
                    if attack_started and close_enough:
                        impact = True
                        impact_t = t_now
                        print(f"\n  🎯 ÇARPIŞMA ONAYLANDI! t={t_now:.1f}s — bağlantı "
                              f"koptu, dron hedefe yakındı (~{last_range:.0f} cm). "
                              f"Vuruş.")
                    elif attack_started:
                        print(f"\n  ⚠ Bağlantı koptu ama dron hedeften UZAKTA "
                              f"(~{last_range:.0f} cm) — sim donması, ÇARPIŞMA "
                              f"SAYILMAZ.")
                    else:
                        print("\n  ✗ Bağlantı taarruz başlamadan koptu — "
                              "koşu bitiriliyor (çarpışma sayılmaz).")
                    break

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

                data_valid = not (tx_raw == 0.0 and ty_raw == 0.0)
                meas_changed = data_valid and (tx_raw != last_tx or ty_raw != last_ty)
                if meas_changed and last_tx is not None:
                    jump = math.hypot(tx_raw - last_tx, ty_raw - last_ty)
                    if jump > 50000:  # 500m'den büyük ani sıçrama = bozuk
                        data_valid = False
                        meas_changed = False

                if meas_changed:
                    tx, ty = tx_raw, ty_raw
                    last_tx, last_ty = tx, ty
                    last_meas_t = t_now
                else:
                    tx = last_tx if last_tx is not None else tx_raw
                    ty = last_ty if last_ty is not None else ty_raw

                data_stale = (last_meas_t is not None
                              and (t_now - last_meas_t) > STALE_TIMEOUT_S)

                # Yükseklik hareketli ortalaması (sıçrama reddi)
                if meas_changed:
                    if locked_alt is None or abs(tz - locked_alt) < ALT_JUMP_REJECT_CM:
                        alt_buffer.append(tz)
                        if len(alt_buffer) > ALT_BUFFER_SIZE:
                            alt_buffer.pop(0)
                        alt_reject_count = 0
                    else:
                        alt_reject_count += 1
                        if alt_reject_count >= ALT_REJECT_PERSIST:
                            alt_buffer = [tz]
                            alt_reject_count = 0
                if alt_buffer:
                    locked_alt = sum(alt_buffer) / len(alt_buffer)
                elif locked_alt is None and data_valid:
                    locked_alt = tz

                if locked_alt is None:
                    time.sleep(DT_LOOP)
                    continue

                target_x_log.append(tx)
                target_y_log.append(ty)
                drone_x_log.append(dx)
                drone_y_log.append(dy)

                # Yörünge bufferı ve RoutePlanner yalnızca yeni ölçümle beslenir
                if meas_changed:
                    orbit_buf_t.append(t_now)
                    orbit_buf_x.append(tx)
                    orbit_buf_y.append(ty)
                    self.planner.add_observation(t_now, tx, ty)

                active_route = master_route if orbit_locked else self.planner.route

                intercept_pt = None
                guide_x, guide_y = tx, ty       # Varsayılan: ham hedef
                range_to_target = None          # öngörülen (lead) hedefe 2B mesafe
                terminal = False
                committing = False

                if active_route.fitted:
                    # Faz birikimi (360° tespiti) — İŞARETLİ toplanır
                    current_phi = active_route.nearest_phi(tx, ty)
                    if last_target_phi is not None:
                        delta = current_phi - last_target_phi
                        if delta > math.pi:
                            delta -= 2 * math.pi
                        elif delta < -math.pi:
                            delta += 2 * math.pi
                        phase_acc += delta
                    last_target_phi = current_phi

                    # 360° (2π) tamamlandı mı? → rota kilidi / yeniden fit
                    if abs(phase_acc) >= 2 * math.pi and len(orbit_buf_t) >= 15:
                        if not route_frozen:
                            try:
                                candidate = EllipseRouteModel()
                                candidate.fit_robust(
                                    np.array(orbit_buf_t),
                                    np.array(orbit_buf_x),
                                    np.array(orbit_buf_y),
                                    outlier_threshold_cm=OUTLIER_THRESHOLD_CM
                                )
                                accept = True
                                if orbit_locked:
                                    center_jump = math.hypot(
                                        candidate.cx - master_route.cx,
                                        candidate.cy - master_route.cy)
                                    if np.sign(candidate.omega) != np.sign(master_route.omega):
                                        accept = False
                                    elif center_jump > OUTLIER_THRESHOLD_CM:
                                        accept = False

                                if accept:
                                    master_route = candidate
                                    orbit_locked = True
                                    active_route = master_route
                                    orbit_count += 1
                                    if orbit_count >= LOCK_STABLE_LAPS:
                                        route_frozen = True
                                    print(f"  ★ Yörünge kilitlendi! (tur #{orbit_count}, "
                                          f"{len(orbit_buf_t)} nokta, "
                                          f"periyot≈{master_route.period:.1f}s"
                                          f"{' — DONDU' if route_frozen else ''})")
                                else:
                                    print("  ⚠ Yeni fit reddedildi (yön/merkez "
                                          "tutarsız) — önceki kilit korunuyor.")
                            except ValueError as e:
                                print(f"  ✗ Robust fit başarısız: {e}")

                        phase_acc = 0.0
                        orbit_buf_t.clear()
                        orbit_buf_x.clear()
                        orbit_buf_y.clear()
                        last_target_phi = active_route.nearest_phi(tx, ty)

                    # --- KAMİKAZE INTERCEPT (ölçüme demirli) ---
                    # ÖNEMLİ: vuruş noktası MODEL SAATİNE (position_at: phi0+omega·t)
                    # göre alınırsa, donmuş rotanın omega'sındaki minik hata
                    # 30-60 s'de faz kayması olarak birikip nişanı gerçek hedeften
                    # ~10 m kaydırıyor → dron hayalet noktanın etrafında dönüyor.
                    # Bunun yerine hedefin GERÇEK anlık konumunu ANLIK ÖLÇÜME
                    # demirliyoruz: ölçülen (tx,ty) elipse izdüşür (gürültü
                    # temizlenir), yalnızca GPS gecikmesi kadar (omega·LEAD)
                    # ileri taşınır. Böylece nişan hep hedefin gerçek yeridir.
                    phi_meas = active_route.nearest_phi(tx, ty)
                    phi_meas_lead = phi_meas + active_route.omega * LEAD_TIME_S
                    # Faz-PLL: omega ile ilerlet + ölçüme doğru küçük düzeltme.
                    # GPS gürültüsünü/OFFSET jitterını süzer, hedefi canlı izler.
                    if phi_track is None:
                        phi_track = phi_meas_lead
                    else:
                        phi_track += active_route.omega * dt
                        err = phi_meas_lead - phi_track
                        while err > math.pi:
                            err -= 2 * math.pi
                        while err < -math.pi:
                            err += 2 * math.pi
                        phi_track += 0.08 * err
                    phi_now = phi_track
                    tgt_now_x, tgt_now_y = active_route.point_on_ellipse(phi_now)
                    range_to_target = math.hypot(tgt_now_x - dx, tgt_now_y - dy)
                    terminal = range_to_target < TERMINAL_RANGE_CM

                    # Faz-demirli tam çarpışma noktası: dron düz uçarsa hedefle
                    # aynı anda buluşacağı nokta. Çözüm çarpışma ROTASI verir —
                    # sonda sert dönüş gerekmez (spin'in kök nedeni buydu).
                    hit_pt = phase_intercept(active_route, phi_now, (dx, dy),
                                             KAMIKAZE_SPEED)
                    if hit_pt is not None:
                        guide_x, guide_y = float(hit_pt[0]), float(hit_pt[1])
                    else:
                        # Yakalanamıyor (ör. hedef önde ve dron yavaş kalıyor):
                        # hedefin gerçek anlık konumuna saf takip (ram).
                        guide_x, guide_y = float(tgt_now_x), float(tgt_now_y)

                    # COMMIT: menzil daralınca yaw yetkisi kısılıp düz dalışa
                    # geçilir (anti-spin). Nişan noktasını LATCH'LEMİYORUZ —
                    # latch, ~2 sn'lik dalışta hedef 3-4 m ilerlediği için
                    # bayatlıyordu. Her döngü intercept yeniden çözülür (hedefi
                    # canlı izler); ölçüm gürültüsü faz-LPF ile yumuşatılmış.
                    committing = range_to_target < COMMIT_RANGE_CM
                    last_range = range_to_target   # vuruş teyidi için (bağlantı kopması)

                    intercept_pt = (float(guide_x), float(guide_y))

                    # Elips çizim verisi
                    if not self.headless:
                        phis_draw = np.linspace(-np.pi, np.pi, 200)
                        ex = active_route.cx + active_route.a * np.cos(phis_draw) * np.cos(active_route.theta) - active_route.b * np.sin(phis_draw) * np.sin(active_route.theta)
                        ey = active_route.cy + active_route.a * np.cos(phis_draw) * np.sin(active_route.theta) + active_route.b * np.sin(phis_draw) * np.cos(active_route.theta)
                        line_fitted_ellipse.set_data(ex, ey)

                t_hist.append(t_now)
                dist_log.append(math.hypot(tx - dx, ty - dy))
                res_log.append(self.planner.last_residual if self.planner.last_residual is not None else 0.0)

                # ====================================================================
                # DRONE YÖNLENDİRME KONTROLCÜSÜ
                # ====================================================================
                # Guide (vuruş) noktası eğim-sınırı: intercept çözümü / nearest_phi
                # sıçramalarını taarruz hızının ~2 katıyla sınırlayıp yumuşatır.
                guide_slew = KAMIKAZE_SPEED * 2.0 * dt
                if last_guide is not None:
                    gdx = guide_x - last_guide[0]
                    gdy = guide_y - last_guide[1]
                    glen = math.hypot(gdx, gdy)
                    if glen > guide_slew:
                        guide_x = last_guide[0] + gdx * guide_slew / glen
                        guide_y = last_guide[1] + gdy * guide_slew / glen
                last_guide = (guide_x, guide_y)

                ldx = guide_x - dx
                ldy = guide_y - dy
                ldz = locked_alt - dz  # Ortalama (eş) irtifaya kilitli
                # Terminal dalış: hedef irtifasının ALTINA nişan al → irtifadan
                # geçerek çarp. Sadece son 15 m'de (commit) uygularsak 17 m/s'de
                # <1 s'de geçip alçalmaya vakit kalmıyordu; bu yüzden tüm terminal
                # menzil (30 m) boyunca, yaklaştıkça derinleşen kademeli dalış.
                if terminal and range_to_target is not None:
                    frac = 1.0 - min(1.0, range_to_target / max(TERMINAL_RANGE_CM, 1.0))
                    ldz -= TERMINAL_DESCENT_CM * frac

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

                # İlk yörünge kilidine kadar BEKLE: rota bilinmeden vuruş noktası
                # çözülemez — hover'da bekle, kilit gelince taarruza geç.
                waiting = WAIT_FOR_LOCK and not orbit_locked
                if waiting and not wait_msg_shown:
                    print("  ⏸ İlk yörünge kilidi bekleniyor — hover'da bekleniyor...")
                    wait_msg_shown = True
                elif not waiting and wait_msg_shown:
                    print("  ▶ Yörünge kilitli — KAMİKAZE taarruza geçiliyor.")
                    wait_msg_shown = False

                # Kilit geldi ve bekleme bitti → taarruz fiilen başladı. Bundan
                # sonra bağlantı kopması VURUŞ olarak onaylanır (yukarıda).
                if orbit_locked and not waiting:
                    attack_started = True

                # ---- YAW: çarpışma noktasına kilitli kerteriz ----
                yaw_error, yaw_cmd = self.yaw_ctrl.calculate(ldx, ldy, d_rot[2], dt)

                if waiting or data_stale:
                    # Bekleme / yeni ölçüm yok: dönme, burnu sabit tut (bayat
                    # konumun etrafında dönmek yerine düz git / committed dalış).
                    yaw_cmd = 0.0
                    self.yaw_ctrl.inner_pid.integral = 0.0
                    self.yaw_ctrl.inner_pid.first_run = True
                    last_yaw_cmd = 0.0
                    pitch_align_err = 0.0
                elif not data_valid:
                    yaw_cmd = last_yaw_cmd
                    pitch_align_err = yaw_error
                else:
                    # COMMIT (anti-spin): hedefe yaklaşınca guide noktasının
                    # üstünden geçerken kerteriz 180° savrulur → dron kendi
                    # etrafında spinler (21 m dönüş yarıçapıyla zaten hedefe
                    # dönemez). Bu menzilde yaw yetkisi kısılır: küçük
                    # düzeltmeye izin var ama spin yok — çarpışma ROTASINDA
                    # (phase_intercept) düz dalıp hedefin içinden geçer.
                    if committing:
                        yaw_cmd = max(-YAW_COMMIT_CAP, min(YAW_COMMIT_CAP, yaw_cmd))
                    last_yaw_cmd = yaw_cmd
                    pitch_align_err = yaw_error

                # ---- İLERİ HIZ: sabit taarruz hızı (kapanma) ----
                v_des = max(MIN_SPEED_CMS, min(CHASE_MAX_CMS, KAMIKAZE_SPEED))
                if waiting:
                    v_des = 0.0     # kilit yok: yerinde bekle (imzalı v_fwd 0'a frenler)

                # ---- PITCH: hız PID → istenen açı → angle-mode stick ----
                # Hız geri beslemesi İMZALI ileri hız (hız vektörünün burun
                # yönüne izdüşümü); skaler |speed| geriye kaçışı gizler.
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

                # ---- THROTTLE: kilitli ortalama (eş) irtifaya PID ----
                thr_cmd = self.throttle_ctrl.calculate(ldz, dt)

                fy = max(-1.0, min(1.0, yaw_cmd))
                fp = -max(-1.0, min(1.0, pitch_cmd))   # sim_bridge işaret düzeni
                f_thr = max(-1.0, min(1.0, thr_cmd))

                drone.set_control_surfaces(f_thr, fp, 0.0, fy, True)

                # ---- VURUŞ TESPİTİ ----
                # Öngörülen (lead) hedefe 3B mesafe eş-irtifa hatası dahil.
                truth = None
                true_dist = -1.0
                if self.headless:
                    truth = drone.get_debug_truth()
                    if truth.get("available"):
                        tp = truth["target"]["position"]
                        true_dist = math.hypot(tp[0] - dx, tp[1] - dy)
                        true_dist = math.hypot(true_dist, tp[2] - dz)
                        min_true_dist = min(min_true_dist, true_dist)

                # Vuruş yarıçapıyla ERKEN BİTİRME YOK — en yakın geçiş yalnızca
                # rapor metriği olarak izlenir; görev bağlantı kopana dek sürer.
                if orbit_locked and not waiting and range_to_target is not None:
                    range_3d = math.hypot(range_to_target, ldz)
                    min_meas_dist = min(min_meas_dist, range_3d)

                if self.headless:
                    log_rows.append((
                        t_now, dx, dy, dz, tx, ty, tz, locked_alt,
                        dist_2d, dist_3d, yaw_error, d_rot[2], yaw_rate_meas,
                        v_fwd, v_des, fy, fp, f_thr, 0.0,
                        int(meas_changed), int(data_stale), int(terminal),
                        int(orbit_locked), true_dist,
                    ))

                if self.duration is not None and t_now >= self.duration:
                    print(f"\n  Süre doldu ({self.duration:.0f} s) — koşu bitiriliyor.")
                    break

                # Arayüzü güncelle
                step += 1
                if not self.headless and step % 5 == 0 and target_x_log:
                    line_target_path.set_data(target_x_log, target_y_log)
                    line_drone_path.set_data(drone_x_log, drone_y_log)
                    point_target.set_data([tx], [ty])
                    point_drone.set_data([dx], [dy])

                    if intercept_pt is not None:
                        point_intercept.set_data([intercept_pt[0]], [intercept_pt[1]])
                        line_intercept_vector.set_data([dx, intercept_pt[0]], [dy, intercept_pt[1]])
                    else:
                        point_intercept.set_data([], [])
                        line_intercept_vector.set_data([], [])

                    ax_map.relim(); ax_map.autoscale_view()
                    ax_map.set_aspect('equal', 'datalim')

                    line_dist.set_data(t_hist, dist_log)
                    ax_dist.relim(); ax_dist.autoscale_view()

                    line_res.set_data(t_hist, res_log)
                    ax_res.relim(); ax_res.autoscale_view()

                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

                elapsed = time.time() - step_start
                sleep_t = DT_LOOP - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            print("\n  Canlı taarruz sonlandırıldı.")
        finally:
            try:
                drone.set_control_surfaces(0.0, 0.0, 0.0, 0.0, True)
            except Exception:
                pass
            drone.disconnect()
            if self.headless:
                self._write_report(log_rows, impact, impact_t,
                                   min_true_dist, min_meas_dist, HIT_RADIUS_CM)
            else:
                plt.ioff()
                plt.show()

    def _write_report(self, log_rows, impact, impact_t,
                      min_true_dist, min_meas_dist, hit_radius):
        """Headless koşu verisini CSV'ye yazar, taarruz özet metriklerini basar
        ve özet grafiği PNG olarak kaydeder."""
        if not log_rows:
            print("  (rapor) Kayıtlı veri yok.")
            return

        cols = ["t", "dx", "dy", "dz", "tx", "ty", "tz", "locked_alt",
                "dist_2d", "dist_3d", "yaw_err", "heading", "yaw_rate",
                "d_speed", "v_des", "cmd_yaw", "cmd_pitch", "cmd_thr", "cmd_roll",
                "meas_changed", "data_stale", "terminal", "orbit_locked",
                "true_dist"]
        arr = np.array(log_rows, dtype=float)
        os.makedirs(self.log_dir, exist_ok=True)
        csv_path = os.path.join(self.log_dir, "route_kamikaze_run.csv")
        np.savetxt(csv_path, arr, delimiter=",", header=",".join(cols), comments="")

        t = arr[:, 0]
        c = {name: arr[:, i] for i, name in enumerate(cols)}

        dur = t[-1] - t[0]
        dh = np.diff(c["heading"])
        dh = (dh + 180.0) % 360.0 - 180.0
        revs = float(np.sum(np.abs(dh)) / 360.0)
        yaw_rate_rms = float(np.sqrt(np.mean(c["yaw_rate"] ** 2)))
        spd = c["d_speed"]
        alt_err = c["locked_alt"] - c["dz"]

        print("\n  " + "=" * 56)
        print("  KAMİKAZE KOŞU RAPORU")
        print(f"    Süre                : {dur:8.1f} s   ({len(t)} örnek)")
        if impact:
            print(f"    SONUÇ               : 🎯 ÇARPIŞMA — bağlantı koptu (t={impact_t:.1f}s)")
        else:
            print(f"    SONUÇ               : ✗ çarpışma yok (bağlantı sürüyor / süre doldu)")
        if min_true_dist < float("inf"):
            print(f"    En yakın geçiş(gerçek): {min_true_dist:8.0f} cm  (vuruş yarıçapı {hit_radius:.0f})")
        if min_meas_dist < float("inf"):
            print(f"    En yakın geçiş(tahmin): {min_meas_dist:8.0f} cm")
        print(f"    Toplam gövde dönüşü : {revs:8.2f} tur (birikimli |Δyaw|/360)")
        print(f"    Yaw hızı RMS        : {yaw_rate_rms:8.1f} deg/s")
        print(f"    Drone hızı          : ort {np.mean(spd):6.0f} / std {np.std(spd):5.0f} cm/s")
        print(f"    İrtifa hatası RMS   : {np.sqrt(np.mean(alt_err**2)):8.0f} cm")
        print(f"    Bayat veri oranı    : {100.0*np.mean(c['data_stale']):8.1f} %")
        print(f"    Terminal faz oranı  : {100.0*np.mean(c['terminal']):8.1f} %")
        print(f"    CSV                 : {csv_path}")

        # --- Özet grafik ---
        plt.switch_backend("Agg")
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        fig.suptitle("Route Kamikaze — Headless Taarruz Raporu", fontweight="bold")

        ax = axes[0, 0]
        ax.plot(c["tx"], c["ty"], ".", ms=2, color="tab:red", alpha=0.4, label="Hedef GPS")
        ax.plot(c["dx"], c["dy"], color="tab:blue", lw=1.2, label="Drone")
        ax.plot(c["dx"][-1], c["dy"][-1], "X", ms=12, color="tab:green"
                if impact else "tab:orange", label="Son konum")
        ax.set_title("XY Yörünge"); ax.set_xlabel("X (cm)"); ax.set_ylabel("Y (cm)")
        ax.axis("equal"); ax.legend(); ax.grid(True, alpha=0.4)

        ax = axes[0, 1]
        ax.plot(t, c["dz"], color="tab:blue", label="Drone Z")
        ax.plot(t, c["tz"], ".", ms=2, color="tab:red", alpha=0.4, label="Hedef Z (ham)")
        ax.plot(t, c["locked_alt"], color="tab:green", label="Kilitli ort. irtifa")
        ax.set_title("İrtifa"); ax.set_xlabel("t (s)"); ax.set_ylabel("Z (cm)")
        ax.legend(); ax.grid(True, alpha=0.4)

        ax = axes[1, 0]
        # Gerçek hedef mesafesi (varsa) — kapanma profili
        td = c["true_dist"].copy()
        td[td < 0] = np.nan
        ax.plot(t, td, color="tab:orange", label="GERÇEK 3B mesafe")
        ax.axhline(hit_radius, color="tab:red", ls="--", lw=1.0, label="Vuruş yarıçapı")
        ax2 = ax.twinx()
        ax2.plot(t, c["d_speed"], color="tab:cyan", lw=0.8, label="Drone hızı")
        ax2.plot(t, c["v_des"], color="tab:green", lw=0.8, ls="--", label="İstenen hız")
        ax.set_title("Kapanma Mesafesi ve Hız"); ax.set_xlabel("t (s)")
        ax.set_ylabel("Mesafe (cm)"); ax2.set_ylabel("Hız (cm/s)")
        ax.legend(loc="upper right"); ax2.legend(loc="lower right")
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
        png_path = os.path.join(self.log_dir, "route_kamikaze_run.png")
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        print(f"    Rapor PNG           : {png_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Route Kamikaze System Testi")
    parser.add_argument("--sim", action="store_true", help="Taarruz simülasyonu çalıştırır.")
    parser.add_argument("--headless", action="store_true",
                        help="Canlı modda GUI açmadan koşar; CSV + rapor PNG üretir.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Canlı koşu süresi (s); dolunca otomatik durur.")
    parser.add_argument("--log-dir", default=None,
                        help="Headless rapor/CSV çıktı dizini (varsayılan: çalışma dizini).")
    args = parser.parse_args()

    if args.sim:
        np.random.seed(42)
        sim = KamikazeSimulator()
        sim.run()
    else:
        runner = LiveKamikazeRunner(headless=args.headless,
                                    duration=args.duration,
                                    log_dir=args.log_dir)
        runner.run()


if __name__ == "__main__":
    main()
