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
from guidance.pid_core import PID, SlewLimiter

# Sabitler
DT_LOOP = 0.01  # Döngü hızı (10 Hz)
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
    def __init__(self):
        self.planner = RoutePlanner(buffer_size=60, min_fit_points=15, residual_threshold=1500.0)
        self._running = True

        # --- Güdüm PID konfigürasyonu (config/pid_params.yaml) ---
        # Ham oransal (P-only) kontrolcü ani tepkiler üretiyordu; buradan
        # okunan kazançlarla yaw/pitch artık türev sönümlemeli (PD) çalışır.
        cfg_path = os.path.join(PROJECT_ROOT, "config", "pid_params.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        ip = self.config["inner_pid"]
        self.yaw_pid = PID(kp=ip["yaw"]["kp"], ki=ip["yaw"]["ki"],
                           kd=ip["yaw"]["kd"], clamp=(-1.0, 1.0))
        self.pitch_pid = PID(kp=ip["pitch"]["kp"], ki=ip["pitch"]["ki"],
                             kd=ip["pitch"]["kd"], clamp=(-1.0, 1.0))
        # Pitch setpoint eğim-sınırlayıcı: faz geçişi / dalış anındaki ani
        # setpoint sıçramalarını (deg/s) sınırlar → "ani tepki" kaynağını keser.
        self.pitch_slew = SlewLimiter(self.config["slew"]["pitch_max_change_per_sec"])

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

        # Kontrolcü Parametreleri — yaw/pitch kazançları artık
        # config/pid_params.yaml üzerinden PID kontrolcülere (self.yaw_pid /
        # self.pitch_pid) taşındı. Burada sadece throttle sabitleri kaldı.
        BASE_HOVER_THR = 0.58
        max_allowed_thr = 1.0
        PITCH_SIGN = -1.0

        # Yörünge Kilitleme Sistemi
        OUTLIER_THRESHOLD_CM = 5000.0   # Elipsten bu mesafeden uzak noktalar outlier
        TRAIL_PHI = 0.35                # ~20 derece arkasında kal
        CROSS_TRACK_BAND = 2000.0       # cm — elipse çekme bandı
        master_route = EllipseRouteModel()
        orbit_locked = False
        route_frozen = False          # yakınsama sonrası rotayı dondur (tur 5-6 titremesini önler)
        LOCK_STABLE_LAPS = 3          # bu kadar tur kilitlendikten sonra yeniden-fit durdurulur
        orbit_buf_t, orbit_buf_x, orbit_buf_y = [], [], []
        phase_acc = 0.0
        last_target_phi = None
        last_guide = None             # guide noktası eğim-sınırı için önceki değer
        orbit_count = 0

        # Yükseklik sabitleme sistemi
        alt_buffer = []              # Hedef Z değerlerinin hareketli ortalaması
        ALT_BUFFER_SIZE = 50         # ~5 saniyelik ortalama (10 Hz'de)
        locked_alt = None            # Kilitli ortalama yükseklik

        # Son komut hafızası (veri kesildiğinde tutulacak)
        last_yaw_cmd = 0.0
        last_tx, last_ty = None, None

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
                tar_pos = t_data["target"]["position"]
                d_speed = t_data["drone"]["speed"]

                dx, dy, dz = d_pos[0], d_pos[1], d_pos[2]
                tx_raw, ty_raw, tz = tar_pos[0], tar_pos[1], tar_pos[2]
                t_speed = t_data["target"].get("speed", 0.0)

                # Konum verisi geçerli mi kontrol et (0,0,0 veya öncekiyle aynı = bozuk)
                data_valid = not (tx_raw == 0.0 and ty_raw == 0.0)
                if data_valid and last_tx is not None:
                    jump = math.hypot(tx_raw - last_tx, ty_raw - last_ty)
                    if jump > 50000:  # 500m'den büyük ani sıçrama = bozuk
                        data_valid = False

                if data_valid:
                    tx, ty = tx_raw, ty_raw
                    last_tx, last_ty = tx, ty
                else:
                    # Bozuk veri — son geçerli konumu kullan
                    tx = last_tx if last_tx is not None else tx_raw
                    ty = last_ty if last_ty is not None else ty_raw

                # Yükseklik ortalaması (ani fırlamayı önle)
                alt_buffer.append(tz)
                if len(alt_buffer) > ALT_BUFFER_SIZE:
                    alt_buffer.pop(0)
                locked_alt = sum(alt_buffer) / len(alt_buffer)

                target_x_log.append(tx)
                target_y_log.append(ty)
                drone_x_log.append(dx)
                drone_y_log.append(dy)

                # Yörünge bufferına ekle
                orbit_buf_t.append(t_now)
                orbit_buf_x.append(tx)
                orbit_buf_y.append(ty)

                # İlk fit için RoutePlanner'ı besle
                self.planner.add_observation(t_now, tx, ty)

                # Aktif rota: kilitliyse master, değilse planner
                active_route = master_route if orbit_locked else self.planner.route

                intercept_pt = None
                guide_x, guide_y = tx, ty  # Varsayılan: ham hedefe git

                if active_route.fitted:
                    # Hedefin elips üzerindeki faz açısını bul
                    current_phi = active_route.nearest_phi(tx, ty)

                    # Faz birikimini izle (360° tespiti)
                    if last_target_phi is not None:
                        delta = current_phi - last_target_phi
                        if delta > math.pi:
                            delta -= 2 * math.pi
                        elif delta < -math.pi:
                            delta += 2 * math.pi
                        phase_acc += abs(delta)
                    last_target_phi = current_phi

                    # 360° (2π) tamamlandı mı?
                    if phase_acc >= 2 * math.pi and len(orbit_buf_t) >= 15:
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
                    target_phi = active_route.nearest_phi(tx, ty)
                    trail_phi = target_phi - np.sign(active_route.omega) * TRAIL_PHI
                    guide_x, guide_y = active_route.point_on_ellipse(trail_phi)

                    # Cross-track düzeltmesi
                    drone_phi = active_route.nearest_phi(dx, dy)
                    near_x, near_y = active_route.point_on_ellipse(drone_phi)
                    cross_track_err = math.hypot(dx - near_x, dy - near_y)

                    if cross_track_err > CROSS_TRACK_BAND:
                        blend = min(1.0, cross_track_err / (CROSS_TRACK_BAND * 3.0))
                        guide_x = guide_x * (1.0 - blend) + near_x * blend
                        guide_y = guide_y * (1.0 - blend) + near_y * blend

                    intercept_pt = (float(guide_x), float(guide_y))

                    # Elips çizim verisi
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
                # sıçramalarını hedef hızının ~2 katıyla sınırlayıp yumuşatır,
                # böylece yaw/pitch komutlarına basamak (step) girişi olmaz.
                guide_slew = max(t_speed, 1500.0) * 2.0 * dt
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

                # Yaw (Sapma) yönelim kontrolü
                target_yaw = math.degrees(math.atan2(ldy, ldx))
                yaw_error = target_yaw - d_rot[2]
                while yaw_error > 180:
                    yaw_error -= 360
                while yaw_error < -180:
                    yaw_error += 360
                
                # Yaw: config PID (P+I+D). D terimi eski ağır EMA'nın (0.7/0.3)
                # faz gecikmesi olmadan salınımı söndürür; hafif EMA yalnızca
                # ölçüm gürültüsünü filtreler.
                final_yaw = self.yaw_pid.calculate(yaw_error, dt)
                if not hasattr(self, "_yaw_smooth"):
                    self._yaw_smooth = 0.0
                self._yaw_smooth = 0.5 * self._yaw_smooth + 0.5 * final_yaw
                final_yaw = self._yaw_smooth

                # Veri geçersizse son yaw komutunda kal
                if not data_valid:
                    final_yaw = last_yaw_cmd
                else:
                    last_yaw_cmd = final_yaw

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

                # Pitch: hedef eğim setpoint'i önce eğim-sınırlayıcıdan geçirilir
                # (faz geçişi / dalış anındaki ani sıçramaları yayar), ardından
                # config PID (P+D) uygulanır — "ani tepki" kaynağı buydu.
                target_pitch_drone = self.pitch_slew.update(target_pitch_drone, dt)
                pitch_error = target_pitch_drone - d_rot[1]
                final_pitch = PITCH_SIGN * self.pitch_pid.calculate(pitch_error, dt)

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
                if step % 5 == 0 and target_x_log:
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
            drone.disconnect()
            plt.ioff()
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Route Intercept System Testi")
    parser.add_argument("--sim", action="store_true", help="Yakalama simülasyonu çalıştırır.")
    args = parser.parse_args()

    if args.sim:
        np.random.seed(42)
        sim = InterceptSimulator()
        sim.run()
    else:
        runner = LiveInterceptRunner()
        runner.run()


if __name__ == "__main__":
    main()
