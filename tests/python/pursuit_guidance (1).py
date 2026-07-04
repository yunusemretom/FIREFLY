"""
PURSUIT GUIDANCE — Surekli Guncellenen Hedef Noktasi
========================================================
SORUN: Drone, 1Hz'de gelen GPS/UKF noktasina varip orada "bekliyor",
       cunku yonlendirme komutu sadece yeni GPS geldiginde guncelleniyor.

COZUM: Rehberlik hedefi, HER kontrol tick'inde (drone_sdk'nin komut
       hizinda, orn. 10Hz), tracker'in kisa-ufuklu (lead-time) ONE
       BAKISLI tahminiyle surekli guncellenir. Boylece drone sabit bir
       noktaya varip beklemek yerine, hedefin BIRAZ ILERISINDEKI
       (surekli hareket eden) noktaya dogru kesintisiz yonelir.

NEDEN BU CALISIR:
  - tracker.predict_only(t) yan-etkisiz (filtrenin gercek durumunu
    degistirmez, sadece ekstrapole eder) -> guvenle her tick'te
    cagrilabilir.
  - Lead-time KISA tutulmali (0.5-2s). Cunku CTRV'nin kendi
    ekstrapolasyonu birkaç saniyeden sonra hizla sapar (bkz. onceki
    analiz: 1s->~2400 birim hata, 5s->~7900, 10s->~13900). Ama since
    her GPS ölçümü (1Hz) filtreyi surekli gercek konuma "cektigi" icin,
    KISA lead-time'li surekli tahmin, TEK SEFERLIK uzun ekstrapolasyondan
    cok daha guvenlidir -- her saniye "sifirlanip" yeniden baslar.
  - Sonuc: drone her zaman hedefin ~lead_time saniye sonraki tahmini
    konumuna dogru gider, hedef GPS noktasina ulastiginda "beklemek"
    yerine kesintisiz takip eder.

KULLANIM (mevcut LiveTrackerRunner icine entegrasyon):
    guidance = PursuitGuidance(tracker, lead_time=1.0)
    ...
    while running:
        t_now = ...
        if gps_geldi:
            tracker.update(t_now, x, y, z)
        else:
            tracker.predict_only(t_now)   # (tracker'in kendi ic durumunu ilerletmek icin)

        # HER TICK'TE (GPS gelsin gelmesin) komut gonder:
        cmd = guidance.compute_command(t_now, drone_pos, drone_max_speed)
        drone.set_velocity(cmd['velocity_vector'])   # ya da drone_sdk'nizin uygun metodu
"""

import numpy as np
import time
import math
class PursuitGuidance:
    def __init__(self, tracker, lead_time=1.0, min_lead=0.3, max_lead=2.5):
        """
        tracker    : FixedWingTracker orneği (zaten update()/predict_only()
                     ile calisir durumda olmali)
        lead_time  : hedefin kac saniye ilerisine dogru yonelinecegi.
                     Kisa tutulmali (0.5-2s onerilir); GPS periyoduna
                     (1s) yakin veya biraz uzeri iyi bir baslangic noktasidir.
        """
        self.tracker = tracker
        self.lead_time = float(np.clip(lead_time, min_lead, max_lead))
        self.min_lead = min_lead
        self.max_lead = max_lead

    def get_lead_point(self, t_now):
        """Su an + lead_time saniye sonraki tahmini hedef konumunu dondurur.
        Tracker'in kendi durumunu DEGISTIRMEZ (predict_only yan-etkisiz)."""
        result = self.tracker.predict_only(t_now + self.lead_time)
        return result['position'], result

    def compute_command(self, t_now, drone_pos, drone_max_speed,
                         arrival_slowdown_radius=None):
        """
        drone_pos          : (x,y,z) dronun su anki konumu
        drone_max_speed    : dronun maksimum hizi (birim/s)
        arrival_slowdown_radius : bu mesafenin altinda hiz kademeli dusurulur
                                   (opsiyonel, hedefe cok yaklasinca "asiri
                                   sapma"yi/overshoot'u onlemek icin)

        Donen: {'velocity_vector', 'target_point', 'distance', 'heading_deg'}
        """
        (tx, ty, tz), _ = self.get_lead_point(t_now)
        dx, dy, dz = drone_pos
        vec = np.array([tx - dx, ty - dy, tz - dz])
        dist = np.linalg.norm(vec)

        if dist < 1e-6:
            direction = np.zeros(3)
        else:
            direction = vec / dist

        speed = drone_max_speed
        if arrival_slowdown_radius and dist < arrival_slowdown_radius:
            speed = drone_max_speed * max(dist / arrival_slowdown_radius, 0.15)

        velocity_vector = direction * speed
        heading_deg = np.degrees(np.arctan2(direction[1], direction[0])) % 360

        return {
            'velocity_vector': tuple(velocity_vector),
            'target_point': (tx, ty, tz),
            'distance': float(dist),
            'heading_deg': float(heading_deg),
        }

    def adapt_lead_time(self, tracker_speed_estimate, gps_period=1.0):
        """
        Istege bagli: hedefin tahmini hizina gore lead_time'i otomatik ayarlar.
        Hizli/az manevrali hedeflerde daha uzun, yavas/cok donuslu hedeflerde
        daha kisa lead_time daha guvenlidir. Basit bir sezgisel kural:
        lead_time ~ gps_period (1 GPS periyodu ileriye bakmak cogu durumda
        iyi bir denge noktasidir).
        """
        self.lead_time = float(np.clip(gps_period, self.min_lead, self.max_lead))
        return self.lead_time


# ============================================================================
# DUAL-MODE ÇALIŞTIRMA SİSTEMİ
# ============================================================================
class PursuitSimulator:
    def __init__(self):
        np.random.seed(1)
        # basit dairesel hedef simulasyonu (SIM_SPEED/SIM_RADIUS benzeri)
        self.SIM_SPEED, self.SIM_RADIUS = 1800.0, 25000.0
        self.omega = self.SIM_SPEED / self.SIM_RADIUS
        self.dt_ctrl = 0.1
        self.T = 20.0
        self.tracker = FixedWingTracker()
        self.guidance = PursuitGuidance(self.tracker, lead_time=1.0)
        self.drone_pos = np.array([0.0, -30000.0, 8000.0])
        self.drone_speed = 2200.0  # hedeften biraz hizli, yakalayabilsin

    def run(self):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # Premium Koyu Tema Ayarları
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
            "Pursuit Guidance — Sürekli Öngörülü Takip Simülasyonu",
            fontsize=14, fontweight="bold", color="#e8ecf8", y=0.96
        )

        gs = gridspec.GridSpec(2, 2, figure=fig, left=0.07, right=0.96,
                               top=0.88, bottom=0.08, hspace=0.32, wspace=0.22)

        ax_map = fig.add_subplot(gs[:, 0])    # 2D Takip Haritası (Sol)
        ax_dist = fig.add_subplot(gs[0, 1])   # Hata/Mesafe (Sağ Üst)
        ax_speed = fig.add_subplot(gs[1, 1])  # Hızlar (Sağ Alt)

        # Çizim Elemanları
        line_target_real, = ax_map.plot([], [], color="#29b6f6", lw=1.5, label="Hedef Gerçek Yolu")
        line_target_noisy, = ax_map.plot([], [], 'o', color="#ff5252", ms=4, alpha=0.5, label="Noisy GPS")
        line_drone_path, = ax_map.plot([], [], color="#00e676", lw=2, label="Dron Takip Yolu")

        point_target, = ax_map.plot([], [], 'o', color="#ff1744", ms=8, label="Hedef (Şu anki)")
        point_lead, = ax_map.plot([], [], '*', color="#ffab40", ms=12, label="Öncü Hedef (Lead Point)")
        point_drone, = ax_map.plot([], [], '^', color="#00e5ff", ms=8, label="Takipçi Dron")
        line_lead_vector, = ax_map.plot([], [], '--', color="#ffab40", alpha=0.7)

        ax_map.set_xlabel("X (cm)")
        ax_map.set_ylabel("Y (cm)")
        ax_map.set_title("2D Takip ve Öncü Hedef Planlaması", fontsize=11, fontweight="bold")
        ax_map.legend(loc="upper left", facecolor="#1a1d27", edgecolor="#3a3d4d")
        ax_map.grid(True)

        line_dist, = ax_dist.plot([], [], color="#e040fb", lw=1.5, label="Takip Mesafesi")
        ax_dist.set_xlabel("Zaman (s)")
        ax_dist.set_ylabel("Mesafe (cm)")
        ax_dist.set_title("Dron-Hedef Arası Anlık Mesafe", fontsize=10, fontweight="bold")
        ax_dist.grid(True)

        line_target_speed, = ax_speed.plot([], [], color="#29b6f6", lw=1.5, label="Hedef Tahmini Hız")
        line_drone_speed, = ax_speed.plot([], [], color="#00e676", lw=1.5, label="Dron Hızı")
        ax_speed.set_xlabel("Zaman (s)")
        ax_speed.set_ylabel("Hız (cm/s)")
        ax_speed.set_title("Hız Profili", fontsize=10, fontweight="bold")
        ax_speed.legend(loc="upper right", facecolor="#1a1d27", edgecolor="#3a3d4d")
        ax_speed.grid(True)

        gps_counter = 0.0
        t = 0.0
        step = 0

        t_hist = []
        target_real_x, target_real_y = [], []
        target_noisy_x, target_noisy_y = [], []
        drone_x_log, drone_y_log = [self.drone_pos[0]], [self.drone_pos[1]]
        dist_log = []
        target_speeds, drone_speeds = [], []

        print(f"{'t':>5} | {'drone_pos':>28} | {'lead_target':>28} | {'mesafe':>8}")
        while t < self.T:
            rx = self.SIM_RADIUS * np.cos(self.omega * t)
            ry = self.SIM_RADIUS * np.sin(self.omega * t)
            rz = 8000.0

            gps_counter += self.dt_ctrl
            if gps_counter >= 1.0 - 1e-9 or not self.tracker._initialized:
                gps_counter = 0.0
                nx = rx + np.random.normal(0, 1500)
                ny = ry + np.random.normal(0, 1500)
                nz = rz + np.random.normal(0, 400)
                res = self.tracker.update(t, nx, ny, nz)
                target_noisy_x.append(nx)
                target_noisy_y.append(ny)
                t_speed = res['speed']
            else:
                res = self.tracker.predict_only(t)
                t_speed = res['speed']

            # YENI YONTEM: her tick'te surekli guncellenen lead-point'e dogru git
            cmd = self.guidance.compute_command(t, tuple(self.drone_pos), self.drone_speed)
            self.drone_pos = self.drone_pos + np.array(cmd['velocity_vector']) * self.dt_ctrl

            t_hist.append(t)
            target_real_x.append(rx)
            target_real_y.append(ry)
            drone_x_log.append(self.drone_pos[0])
            drone_y_log.append(self.drone_pos[1])
            dist_log.append(cmd['distance'])
            target_speeds.append(t_speed)
            drone_speeds.append(np.linalg.norm(cmd['velocity_vector']))

            # Ekranı güncelle
            step += 1
            if step % 5 == 0:
                line_target_real.set_data(target_real_x, target_real_y)
                line_target_noisy.set_data(target_noisy_x, target_noisy_y)
                line_drone_path.set_data(drone_x_log, drone_y_log)

                point_target.set_data([rx], [ry])
                point_lead.set_data([cmd['target_point'][0]], [cmd['target_point'][1]])
                point_drone.set_data([self.drone_pos[0]], [self.drone_pos[1]])
                line_lead_vector.set_data([self.drone_pos[0], cmd['target_point'][0]], [self.drone_pos[1], cmd['target_point'][1]])

                ax_map.relim(); ax_map.autoscale_view()
                ax_map.set_aspect('equal', 'datalim')

                line_dist.set_data(t_hist, dist_log)
                ax_dist.relim(); ax_dist.autoscale_view()

                line_target_speed.set_data(t_hist, target_speeds)
                line_drone_speed.set_data(t_hist, drone_speeds)
                ax_speed.relim(); ax_speed.autoscale_view()

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)

            if t % 1.0 < self.dt_ctrl/2:
                print(f"{t:5.1f} | {str(tuple(np.round(self.drone_pos,0))):>28} | "
                      f"{str(tuple(np.round(cmd['target_point'],0))):>28} | {cmd['distance']:8.1f}")

            t += self.dt_ctrl

        plt.ioff()
        print("\nSimülasyon tamamlandı. Grafik penceresini kapatarak çıkabilirsiniz.")
        plt.show()


class LivePursuitRunner:
    def __init__(self):
        self.tracker = FixedWingTracker()
        self.guidance = PursuitGuidance(self.tracker, lead_time=1.0)
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

        print("  Canlı Öngörülü Takip İzleyicisi Başlatıldı. CTRL+C ile durdurun.\n")

        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # Premium Koyu Tema Ayarları
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
            "Pursuit Guidance — Canlı drone_sdk Telemetri Takibi",
            fontsize=14, fontweight="bold", color="#e8ecf8", y=0.96
        )

        gs = gridspec.GridSpec(2, 2, figure=fig, left=0.07, right=0.96,
                               top=0.88, bottom=0.08, hspace=0.32, wspace=0.22)

        ax_map = fig.add_subplot(gs[:, 0])
        ax_dist = fig.add_subplot(gs[0, 1])
        ax_speed = fig.add_subplot(gs[1, 1])

        # Çizim Elemanları
        line_target_real, = ax_map.plot([], [], color="#29b6f6", lw=1.5, label="Hedef Gerçek (Truth)")
        line_target_noisy, = ax_map.plot([], [], 'o', color="#ff5252", ms=4, alpha=0.5, label="Noisy GPS")
        line_drone_path, = ax_map.plot([], [], color="#00e676", lw=2, label="Dron Gerçek Yolu")

        point_target, = ax_map.plot([], [], 'o', color="#ff1744", ms=8, label="Hedef (Canlı)")
        point_lead, = ax_map.plot([], [], '*', color="#ffab40", ms=12, label="Öncü Hedef (Lead Point)")
        point_drone, = ax_map.plot([], [], '^', color="#00e5ff", ms=8, label="Dron")
        line_lead_vector, = ax_map.plot([], [], '--', color="#ffab40", alpha=0.7)

        ax_map.set_xlabel("X (cm)")
        ax_map.set_ylabel("Y (cm)")
        ax_map.set_title("Canlı Konum ve Öngörülü Takip Haritası", fontsize=11, fontweight="bold")
        ax_map.legend(loc="upper left", facecolor="#1a1d27", edgecolor="#3a3d4d")
        ax_map.grid(True)

        line_dist, = ax_dist.plot([], [], color="#e040fb", lw=1.5, label="Hedef-Dron Mesafesi")
        ax_dist.set_xlabel("Zaman (s)")
        ax_dist.set_ylabel("Mesafe (cm)")
        ax_dist.set_title("Hedef ve Dron Arasındaki Mesafe", fontsize=10, fontweight="bold")
        ax_dist.grid(True)

        line_target_speed, = ax_speed.plot([], [], color="#29b6f6", lw=1.5, label="Hedef Tahmini Hız")
        line_drone_speed, = ax_speed.plot([], [], color="#00e676", lw=1.5, label="Dron Hızı")
        ax_speed.set_xlabel("Zaman (s)")
        ax_speed.set_ylabel("Hız (cm/s)")
        ax_speed.set_title("Canlı Hız Profilleri", fontsize=10, fontweight="bold")
        ax_speed.legend(loc="upper right", facecolor="#1a1d27", edgecolor="#3a3d4d")
        ax_speed.grid(True)

        t_hist = []
        target_real_x, target_real_y = [], []
        target_noisy_x, target_noisy_y = [], []
        drone_x_log, drone_y_log = [], []
        dist_log = []
        target_speeds, drone_speeds = [], []

        # Kontrolcü Parametreleri (drone_gui.py ile birebir uyumlu)
        KP_YAW = 0.07
        KP_PITCH = 0.035
        BASE_HOVER_THR = 0.58
        max_allowed_thr = 1.0
        PITCH_SIGN = -1.0

        t0 = time.time()
        gps_counter = 0.0
        step = 0
        dt_ctrl = 0.1
        gps_period = 1.0

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
                tx, ty, tz = tar_pos[0], tar_pos[1], tar_pos[2]

                # GPS güncellemesi (1 Hz) veya tahmin adımı
                gps_counter += dt_ctrl
                if gps_counter >= gps_period - 1e-9 or not self.tracker._initialized:
                    gps_counter = 0.0
                    res = self.tracker.update(t_now, tx, ty, tz)
                    target_noisy_x.append(tx)
                    target_noisy_y.append(ty)
                    t_speed = res['speed']

                    # Debug/Ground truth verisi al (varsa)
                    try:
                        truth = drone.get_debug_truth()
                    except Exception:
                        truth = {}
                    
                    if truth.get("available"):
                        rx = truth["target"]["position"][0]
                        ry = truth["target"]["position"][1]
                    else:
                        rx, ry = np.nan, np.nan
                    target_real_x.append(rx)
                    target_real_y.append(ry)
                else:
                    res = self.tracker.predict_only(t_now)
                    t_speed = res['speed']

                # Pursuit guidance öngörülü komutu hesapla
                cmd = self.guidance.compute_command(t_now, (dx, dy, dz), d_speed)

                # ====================================================================
                # DRONE YÖNLENDİRME KONTROLCÜSÜ (drone_gui.py mantığı)
                # ====================================================================
                lead_pos = cmd['target_point']
                ldx = lead_pos[0] - dx
                ldy = lead_pos[1] - dy
                ldz = lead_pos[2] - dz

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

                # Telemetri veri logları
                t_hist.append(t_now)
                drone_x_log.append(dx)
                drone_y_log.append(dy)
                dist_log.append(cmd['distance'])
                target_speeds.append(t_speed)
                drone_speeds.append(d_speed)

                # Arayüzü 2 Hz hızda güncelle (her 5 adımda bir)
                step += 1
                if step % 5 == 0 and target_noisy_x:
                    line_target_real.set_data(target_real_x, target_real_y)
                    line_target_noisy.set_data(target_noisy_x, target_noisy_y)
                    line_drone_path.set_data(drone_x_log, drone_y_log)

                    point_target.set_data([tx], [ty])
                    point_lead.set_data([cmd['target_point'][0]], [cmd['target_point'][1]])
                    point_drone.set_data([dx], [dy])
                    line_lead_vector.set_data([dx, cmd['target_point'][0]], [dy, cmd['target_point'][1]])

                    ax_map.relim(); ax_map.autoscale_view()
                    ax_map.set_aspect('equal', 'datalim')

                    line_dist.set_data(t_hist, dist_log)
                    ax_dist.relim(); ax_dist.autoscale_view()

                    line_target_speed.set_data(t_hist, target_speeds)
                    line_drone_speed.set_data(t_hist, drone_speeds)
                    ax_speed.relim(); ax_speed.autoscale_view()

                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

                elapsed = time.time() - step_start
                sleep_t = dt_ctrl - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            print("\n  Canlı takip sonlandırıldı.")
        finally:
            drone.disconnect()
            plt.ioff()
            plt.show()


if __name__ == "__main__":
    import argparse
    import sys, os
    sys.path.insert(0, '.')
    from src.python.navigation.fixed_wing_tracker import FixedWingTracker

    parser = argparse.ArgumentParser(description="Pursuit Guidance Testi")
    parser.add_argument("--sim", action="store_true", help="Yakalama simülasyonunu çalıştırır.")
    args = parser.parse_args()

    if args.sim:
        sim = PursuitSimulator()
        sim.run()
    else:
        runner = LivePursuitRunner()
        runner.run()
