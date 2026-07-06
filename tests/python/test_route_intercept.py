"""
================================================================================
TEST: Route Intercept System — Canlı SDK & İnteraktif Yakalama Simülasyonu
================================================================================
Dosya: tests/python/test_route_intercept.py

Çalıştırma:
    python tests/python/test_route_intercept.py [--sim]

Mod seçimi:
    --sim bayrağı yoksa: drone_sdk üzerinden canlı telemetri verisi okunur.
    --sim bayrağı varsa: Dronun elips yörüngede uçan hedefi yakalama simülasyonu çalıştırılır.
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
DRONE_SPEED_CMS = 2500.0  # Dron hızı: 25 m/s

# Simülasyon Ayarları
SIM_TARGET_SPEED = 1800.0  # cm/s (18 m/s)
SIM_CENTER_X = 20000.0     # cm (200 m)
SIM_CENTER_Y = 20000.0     # cm (200 m)
SIM_A = 15000.0            # cm (150 m)
SIM_B = 10000.0            # cm (100 m)
SIM_GPS_NOISE = 500.0      # cm


# ===========================================================================
# SİMÜLASYON MODU
# ===========================================================================
class InterceptSimulator:
    def __init__(self):
        self.planner = RoutePlanner(buffer_size=60, min_fit_points=15, residual_threshold=1500.0)
        self.omega = SIM_TARGET_SPEED / math.sqrt((SIM_A**2 + SIM_B**2)/2.0)

    def run(self):
        print("=" * 60)
        print("  Route Intercept — Yakalama Simülasyonu")
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
            "Route Intercept System — Otonom Yakalama Simülasyonu",
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
        point_intercept, = ax_map.plot([], [], '*', color="#00e676", ms=12, label="Yakalama Noktası (Intercept)")
        line_intercept_vector, = ax_map.plot([], [], '--', color="#00e676", alpha=0.7)

        ax_map.set_xlabel("X (cm)")
        ax_map.set_ylabel("Y (cm)")
        ax_map.set_title("Otonom Yakalama Haritası", fontsize=11, fontweight="bold")
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
        intercept_resolved = False
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

            # Eğer elips fit edilmişse yakalama çöz
            intercept_pt = None
            if self.planner.route.fitted:
                sol = solve_intercept(self.planner.route, t, (dx, dy), DRONE_SPEED_CMS)
                if sol is not None:
                    intercept_pt = sol['intercept_point']
                    intercept_time = sol['intercept_time']
                    
                    # Dronu yakalama noktasına yönlendir
                    ix, iy = intercept_pt
                    v_dx = ix - dx
                    v_dy = iy - dy
                    v_dist = math.hypot(v_dx, v_dy)
                    
                    if v_dist > 100.0:  # Yakalanana kadar ilerle
                        step_dist = min(DRONE_SPEED_CMS * DT_LOOP, v_dist)
                        dx += (v_dx / v_dist) * step_dist
                        dy += (v_dy / v_dist) * step_dist
                        
                    if v_dist < 200.0 and not intercept_resolved:
                        print(f"\n  [YAKALANDI!] Hedef t={t:.2f} saniyede elips üzerinde yakalandı.")
                        intercept_resolved = True

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

            t += DT_LOOP

        plt.ioff()
        print("Simülasyon tamamlandı.")
        plt.show()


# ===========================================================================
# CANLI SDK MODU
# ===========================================================================
class LiveInterceptRunner:
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
        #    normalizasyonu (PID-üstüne-PID DEĞİL; SDK çubuğu açı hedefidir)
        #  - ThrottleController: irtifa PID (taban 0 = hover)
        cfg_path = os.path.join(PROJECT_ROOT, "config", "pid_params.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.yaw_ctrl = YawController(self.config)
        self.pitch_ctrl = PitchController(self.config)
        self.throttle_ctrl = ThrottleController(self.config)

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
                "Route Intercept System — Canlı Yakalama Analizi (drone_sdk)",
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
            point_intercept, = ax_map.plot([], [], '*', color="#00e676", ms=12, label="Yakalama Noktası (Intercept)")
            line_intercept_vector, = ax_map.plot([], [], '--', color="#00e676", alpha=0.7)

            ax_map.set_xlabel("X (cm)")
            ax_map.set_ylabel("Y (cm)")
            ax_map.set_title("Canlı Yakalama Haritası", fontsize=11, fontweight="bold")
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

        # Kontrolcü Parametreleri — kazançlar config/pid_params.yaml üzerinden
        # __init__'teki YawController/PitchController/ThrottleController'da.
        #
        # İleri hız güdümü — hedefin hız kanalı GÜVENİLMEZ (bkz. AI_PROJE_OZETI
        # "hız kanalına güvenme"): hız profili mesafeden çizilir, dronun kendi
        # ölçülen hızıyla kapalı döngüde pitch açısına çevrilir.
        ri_cfg = self.config.get("route_intercept", {})
        NOMINAL_TARGET_SPEED = float(ri_cfg.get("nominal_target_speed_cms", 1700.0))
        FOLLOW_DIST_CM = self.config["pursuit"]["follow_distance_cm"]
        CHASE_MAX_CMS = self.config["pursuit"]["max_speed_cms"]
        MIN_SPEED_CMS = self.config["pursuit"]["min_speed_cms"]
        CATCH_GAIN = float(ri_cfg.get("catch_gain", 0.6))
        RING_V_AUTH = float(ri_cfg.get("ring_v_auth_cms", 1100.0))

        # Yakın alan: guide noktasının üstünden geçerken kerteriz açısı
        # anlamsızlaşıp 180° savrulur — 360° kendi etrafında dönüşlerin
        # kök nedeni. Bu yarıçapta dönüş komutu verilmez, burun sabit kalır.
        NEAR_FIELD_CM = 700.0

        # Yörünge Kilitleme / Takip parametreleri (config: route_intercept)
        OUTLIER_THRESHOLD_CM = 5000.0   # Elipsten bu mesafeden uzak noktalar outlier
        LOCK_STABLE_LAPS = int(ri_cfg.get("lock_stable_laps", 3))
        RI_FOLLOW_CM = float(ri_cfg.get("follow_distance_cm", 800.0))
        LEAD_TIME_S = float(ri_cfg.get("lead_time_s", 1.0))
        WAIT_FOR_LOCK = bool(ri_cfg.get("wait_for_lock", True))
        wait_msg_shown = False
        LOOKAHEAD_PHI = float(ri_cfg.get("lookahead_phi", 0.25))
        CROSS_TRACK_BAND = float(ri_cfg.get("cross_track_band_cm", 2000.0))
        master_route = EllipseRouteModel()
        orbit_locked = False
        route_frozen = False          # yakınsama sonrası rotayı dondur (tur 5-6 titremesini önler)
        orbit_buf_t, orbit_buf_x, orbit_buf_y = [], [], []
        phase_acc = 0.0
        last_target_phi = None
        last_guide = None             # guide noktası eğim-sınırı için önceki değer
        orbit_count = 0

        # Yükseklik sabitleme sistemi — hedefin gerçek irtifası değişmediği
        # için Z, ani sıçramaları reddeden bir hareketli ortalamayla tutulur.
        alt_buffer = []               # Kabul edilen hedef Z ölçümleri
        ALT_BUFFER_SIZE = 30          # son ~30 yeni ölçüm (1 Hz veride ~30 s)
        ALT_JUMP_REJECT_CM = 1500.0   # ortalamadan bu kadar sapan Z = sıçrama, yok say
        ALT_REJECT_PERSIST = 8        # bu kadar ardışık ret = irtifa gerçekten değişmiş
        alt_reject_count = 0
        locked_alt = None             # Kilitli ortalama yükseklik

        # Ölçüm tazeliği — SDK yeni paket gelmese de son değeri tutar (stale
        # hold); konum DEĞİŞMEDİKÇE yeni ölçüm yok demektir. Uzun süre yeni
        # ölçüm gelmezse bayat konum etrafında dönmek yerine düz uçuşa geçilir.
        STALE_TIMEOUT_S = 2.5         # 1 Hz nominal veri için güvenli eşik
        last_meas_t = None            # son GERÇEK (değişen) ölçümün zamanı

        # Son komut hafızası (veri kesildiğinde tutulacak)
        last_yaw_cmd = 0.0
        last_tx, last_ty = None, None
        prev_heading = None           # ölçülen yaw hızı (rapor metriği) için
        arc_filt = None               # yuvaya yay farkı LPF durumu (hız kanalı)
        log_rows = []                 # headless koşu veri kaydı

        t0 = time.time()
        prev_loop_t = t0
        step = 0

        try:
            while self._running:
                step_start = time.time()
                t_now = step_start - t0
                # Gerçek döngü süresi (PID/slew türev ve entegrasyonu için)
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

                # Bağlantı başında telemetri henüz akmıyor olabilir — komut
                # üretmeden bekle (aksi halde sıfır konumlar kontrolcüye girer).
                if dx == 0.0 and dy == 0.0 and tx_raw == 0.0 and ty_raw == 0.0:
                    time.sleep(DT_LOOP)
                    continue

                # Konum verisi geçerli mi kontrol et (0,0,0 = bozuk)
                data_valid = not (tx_raw == 0.0 and ty_raw == 0.0)
                # Değişim tespiti: değer öncekiyle aynıysa SDK bayat veriyi
                # tutuyordur — yeni ölçüm olarak İŞLENMEZ.
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
                    # Bayat/bozuk veri — son geçerli konumu kullan
                    tx = last_tx if last_tx is not None else tx_raw
                    ty = last_ty if last_ty is not None else ty_raw

                # Uzun süredir yeni ölçüm yoksa bayat konuma güdüm yapılmaz
                data_stale = (last_meas_t is not None
                              and (t_now - last_meas_t) > STALE_TIMEOUT_S)

                # Yükseklik: yalnızca YENİ ölçümler ortalamaya girer; kilitli
                # ortalamadan aşırı sapan Z sıçramaları yok sayılır. Ret üst
                # üste sürerse irtifa gerçekten değişmiştir → ortalama tazelenir.
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
                    # İlk geçerli ölçüme kadar kilitlenme YOK; geçersiz (0) Z
                    # ile başlatmak throttle'ı -1'e kilitliyordu.
                    locked_alt = tz

                if locked_alt is None:
                    # Henüz geçerli hedef irtifası yok — komut üretme
                    time.sleep(DT_LOOP)
                    continue

                target_x_log.append(tx)
                target_y_log.append(ty)
                drone_x_log.append(dx)
                drone_y_log.append(dy)

                # Yörünge bufferı ve RoutePlanner yalnızca yeni ölçümle beslenir;
                # bayat verinin her döngüde tekrar eklenmesi fit'i o noktaya çeker.
                if meas_changed:
                    orbit_buf_t.append(t_now)
                    orbit_buf_x.append(tx)
                    orbit_buf_y.append(ty)
                    self.planner.add_observation(t_now, tx, ty)

                # Aktif rota: kilitliyse master, değilse planner
                active_route = master_route if orbit_locked else self.planner.route

                intercept_pt = None
                guide_x, guide_y = tx, ty  # Varsayılan: ham hedefe git
                ring_v_des = None          # halka üstü faz-farkı hız talebi

                if active_route.fitted:
                    # Hedefin elips üzerindeki faz açısını bul
                    current_phi = active_route.nearest_phi(tx, ty)

                    # Faz birikimini izle (360° tespiti). İŞARETLİ toplanır:
                    # |delta| toplamak gürültü jitter'ını da sayıp yarım turda
                    # sahte "tur tamamlandı" üretiyordu (erken/yanlış kilit).
                    if last_target_phi is not None:
                        delta = current_phi - last_target_phi
                        if delta > math.pi:
                            delta -= 2 * math.pi
                        elif delta < -math.pi:
                            delta += 2 * math.pi
                        phase_acc += delta
                    last_target_phi = current_phi

                    # 360° (2π) tamamlandı mı?
                    if abs(phase_acc) >= 2 * math.pi and len(orbit_buf_t) >= 15:
                        # Rota donmadıysa yeniden fit dene; donduysa yalnızca
                        # buffer/faz sıfırlanır. Her turda kör re-fit yapmak elips
                        # parametrelerini oynatıp guide noktasını basamaklı
                        # zıplatıyor (ve omega işaretini çevirip trail'i öne
                        # düşürüyor) — tur 5-6 titremesinin kök nedeni budur.
                        if not route_frozen:
                            try:
                                candidate = EllipseRouteModel()
                                candidate.fit_robust(
                                    np.array(orbit_buf_t),
                                    np.array(orbit_buf_x),
                                    np.array(orbit_buf_y),
                                    outlier_threshold_cm=OUTLIER_THRESHOLD_CM
                                )
                                # Dejenerelik kontrolü: dönüş yönü (omega işareti)
                                # ters dönmemeli, merkez aşırı sıçramamalı; aksi
                                # halde yeni fit reddedilip eski kilit korunur.
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

                        # Sonraki tur için sıfırla (donmuş olsa da buffer büyümesin)
                        phase_acc = 0.0
                        orbit_buf_t.clear()
                        orbit_buf_x.clear()
                        orbit_buf_y.clear()
                        last_target_phi = active_route.nearest_phi(tx, ty)

                    # --- ELİPS ÜZERİNDE TAKİP ---
                    # Dümen (carrot) ile hız (yuvaya yetişme) AYRIŞTIRILIR:
                    # hedefin arkasındaki sabit yuva noktasını doğrudan kovalamak,
                    # noktaya varınca kerteriz çöküşü → 360° tur üretiyordu.
                    dir_sign = 1.0 if active_route.omega >= 0 else -1.0
                    r_eff = 0.5 * (active_route.a + active_route.b)
                    # Takip yuvası cm cinsinden konfigürasyondan faza çevrilir;
                    # eski sabit 0.35 rad, yarıçapla çarpınca ~35 m geride
                    # kalmak demekti ("uzak kalma"nın nedeni).
                    trail_phi = min(0.6, max(0.03, RI_FOLLOW_CM / max(r_eff, 1.0)))
                    target_phi = active_route.nearest_phi(tx, ty)
                    # GPS DELAY bayrağı ölçümü ~1 sn geriden gösterir (20 m/s
                    # hedefte ~20 m!) — hedef fazı rota açısal hızıyla öne
                    # alınarak telafi edilir ("uzak kalma"nın asıl nedeni).
                    target_phi += active_route.omega * LEAD_TIME_S
                    slot_phi = target_phi - dir_sign * trail_phi  # takip yuvası

                    drone_phi = active_route.nearest_phi(dx, dy)
                    near_x, near_y = active_route.point_on_ellipse(drone_phi)
                    cross_track_err = math.hypot(dx - near_x, dy - near_y)

                    if cross_track_err <= CROSS_TRACK_BAND:
                        # HALKA ÜZERİNDE — pure pursuit: carrot daima dronun
                        # kendi fazının İLERİSİNDE, kerteriz asla çökmez.
                        carrot_phi = drone_phi + dir_sign * LOOKAHEAD_PHI
                        gx, gy = active_route.point_on_ellipse(carrot_phi)
                        guide_x, guide_y = float(gx), float(gy)

                        # Yetişme, hız kanalından: yuvaya işaretli faz farkı →
                        # yay mesafesi → nominal üstü/altı hız.
                        gap = slot_phi - drone_phi
                        while gap > math.pi:
                            gap -= 2 * math.pi
                        while gap < -math.pi:
                            gap += 2 * math.pi
                        ahead_arc = dir_sign * gap * r_eff
                        # Yuva fazı 1 Hz GPS gürültüsüyle titrer; yakın takipte
                        # bu doğrudan hız talebine yansır → ~1 sn LPF (yalnızca
                        # hız kanalı; dümen carrot'u drone konumuna bağlı, temiz).
                        if arc_filt is None:
                            arc_filt = ahead_arc
                        arc_filt += 0.02 * (ahead_arc - arc_filt)
                        # Halka taban hızı rotanın kendisinden: |omega|·r_eff =
                        # hedefin ölçülmüş ortalama yol hızı. Sabit varsayım
                        # gerçek hızdan düşük kalınca P dengesi kalıcı gerilik
                        # bırakıyordu ("uzak kalma"nın ikinci nedeni).
                        v_route = abs(active_route.omega) * r_eff
                        if not (500.0 <= v_route <= CHASE_MAX_CMS):
                            v_route = NOMINAL_TARGET_SPEED
                        # Hız yetkisi sınırlı tutulur: sınırsız P, faz farkında
                        # testere-dişi limit cycle (hızlan/yavaşla) üretiyor.
                        catch = max(-RING_V_AUTH, min(RING_V_AUTH, CATCH_GAIN * arc_filt))
                        ring_v_des = v_route + catch
                    else:
                        # HALKAYA YAKLAŞMA: yuva noktasına git; sapma büyüdükçe
                        # halkanın en yakın noktasına doğru sürekli blend.
                        gx, gy = active_route.point_on_ellipse(slot_phi)
                        guide_x, guide_y = float(gx), float(gy)
                        excess = cross_track_err - CROSS_TRACK_BAND
                        blend = min(1.0, excess / (CROSS_TRACK_BAND * 2.0))
                        guide_x = guide_x * (1.0 - blend) + near_x * blend
                        guide_y = guide_y * (1.0 - blend) + near_y * blend

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
                # DRONE YÖNLENDİRME KONTROLCÜSÜ (drone_gui.py mantığı)
                # ====================================================================
                # Guide noktası eğim-sınırı: rota yeniden-fit / nearest_phi
                # sıçramalarını nominal hedef hızının ~2 katıyla sınırlayıp
                # yumuşatır, böylece yaw/pitch komutlarına basamak girişi olmaz.
                guide_slew = NOMINAL_TARGET_SPEED * 2.0 * dt
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
                ldz = locked_alt - dz  # Ortalama yüksekliğe kilitli

                dist_2d = math.hypot(ldx, ldy)
                dist_3d = math.hypot(dist_2d, ldz)

                # Ölçülen yaw hızı (deg/s) — yalnızca rapor metriği için
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

                # İlk yörünge kilidine kadar BEKLE: rota bilinmeden ham hedefi
                # kovalamak verimsiz ve savruk — hover'da dur, rota irtifasına
                # tırman (throttle zaten locked_alt'a kilitli), kilit gelince git.
                waiting = WAIT_FOR_LOCK and not orbit_locked
                if waiting and not wait_msg_shown:
                    print("  ⏸ İlk yörünge kilidi bekleniyor — hover'da bekleniyor...")
                    wait_msg_shown = True
                elif not waiting and wait_msg_shown:
                    print("  ▶ Yörünge kilitli — takibe geçiliyor.")
                    wait_msg_shown = False

                # ---- YAW: kerteriz hatası → dönüş komutu ----
                yaw_error, yaw_cmd = self.yaw_ctrl.calculate(ldx, ldy, d_rot[2], dt)

                if waiting or data_stale or near_field:
                    # Bekleme / yeni ölçüm yok / guide üstündeyiz: dönme,
                    # burnu sabit tut.
                    yaw_cmd = 0.0
                    self.yaw_ctrl.inner_pid.integral = 0.0
                    self.yaw_ctrl.inner_pid.first_run = True
                    last_yaw_cmd = 0.0
                    pitch_align_err = 0.0   # düz uçuşta pitch tam yetkili
                elif not data_valid:
                    # Tek seferlik bozuk paket — son yaw komutunda kal
                    yaw_cmd = last_yaw_cmd
                    pitch_align_err = yaw_error
                else:
                    last_yaw_cmd = yaw_cmd
                    pitch_align_err = yaw_error

                # ---- İLERİ HIZ PROFİLİ ----
                # Halka üstünde: yuvaya faz farkından (yetişme hız kanalında).
                # Yaklaşma fazında: guide mesafesinden — geride kaldıkça
                # CHASE_MAX'e kadar hızlan, yaklaştıkça nominalin altına in.
                if ring_v_des is not None:
                    v_des = ring_v_des
                else:
                    v_des = NOMINAL_TARGET_SPEED + CATCH_GAIN * (dist_2d - FOLLOW_DIST_CM)
                v_des = max(MIN_SPEED_CMS, min(CHASE_MAX_CMS, v_des))
                if waiting:
                    # Kilit yok: yerinde bekle (imzalı v_fwd ile 0'a frenler)
                    v_des = 0.0
                elif data_stale:
                    # Yeni ölçüm yok: sabit nominal hızla düz ilerlemeye devam
                    v_des = NOMINAL_TARGET_SPEED

                # ---- PITCH: hız PID → istenen açı → angle-mode stick ----
                # PitchController yaw hizasıyla (cos) ölçekler: burun dönük
                # değilken ileri eğim verilmez — hedef etrafında geniş salınımlı
                # turların diğer kaynağı buydu.
                #
                # Hız geri beslemesi İMZALI ileri hızdır (hız vektörünün burun
                # yönüne izdüşümü). Skaler |speed| geri kayışı ayırt edemez:
                # nose-up fren geriye hızlanmayı "hâlâ hızlı" diye okuyup freni
                # kilitliyor ve drone 30+ m/s geriye kaçıyordu.
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

                # Tüm komutlar tek TCP satırında; roll 0 — angle-mode FC yatayı
                # kendisi tutar (sim_bridge ile aynı). Eski 5 ayrı set_* çağrısı
                # ara karelerde throttle/pitch uyumsuzluğuna yol açıyordu.
                drone.set_control_surfaces(f_thr, fp, 0.0, fy, True)

                if self.headless:
                    # (Debug) Gerçek hedef mesafesi — SADECE rapor metriği;
                    # güdümde kullanılmaz (yarışmada bu kanal yok).
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
                        int(orbit_locked), true_dist,
                    ))

                if self.duration is not None and t_now >= self.duration:
                    print(f"\n  Süre doldu ({self.duration:.0f} s) — koşu bitiriliyor.")
                    break

                # Arayüzü güncelle (headless modda çizim yok)
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
            print("\n  Canlı takip sonlandırıldı.")
        finally:
            try:
                # Çıkarken sticks nötr, hover'da bırak
                drone.set_control_surfaces(0.0, 0.0, 0.0, 0.0, True)
            except Exception:
                pass
            drone.disconnect()
            if self.headless:
                self._write_report(log_rows)
            else:
                plt.ioff()
                plt.show()

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
        # Kendi etrafında dönüş: birikimli |Δheading| / 360
        dh = np.diff(c["heading"])
        dh = (dh + 180.0) % 360.0 - 180.0
        revs = float(np.sum(np.abs(dh)) / 360.0)
        # Net (işaretli) dönüş ≈ rotayı izlerken atılan meşru turlar; fazlalık
        # (revs - net) ileri-geri savrulma / kendi etrafında dönme göstergesidir.
        revs_net = float(abs(np.sum(dh)) / 360.0)
        yaw_rate_rms = float(np.sqrt(np.mean(c["yaw_rate"] ** 2)))
        spd = c["d_speed"]
        alt_err = c["locked_alt"] - c["dz"]
        # Kilitten sonraki kararlı takip bölümü (varsa)
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
