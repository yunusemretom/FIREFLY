#!/usr/bin/env python3
"""
RF-DETR + TCTrack + SimBridge Entegre Görsel Kılavuzluk Test Scripti
Açıklama:
    - Hedef görüş alanında değilken EKF/GPS tabanlı konum kılavuzluğu (GPS_GUIDANCE) devrededir.
    - Hedef tespit edilip kilitlendiğinde, kamera tabanlı görsel kılavuzluk (VISION_TRACKING) devreye girer.
    - Hedef kaçtığında veya takip süresi bittiğinde tekrar konum kılavuzluğuna geri dönülür.
    - Tüm parametreler config/pid_params.yaml dosyasından dinamik olarak yüklenir.
"""

import os
import sys
import time
import math
import yaml
from pathlib import Path
import cv2
import numpy as np
import mss

# Proje kök dizinini sys.path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import drone_sdk as drone
from src.python.guidance.mode_manager import ModeManager
from src.python.guidance.pid_pitch import PitchController
from src.python.guidance.pid_throttle import ThrottleController
from src.python.guidance.pid_yaw import YawController
from src.python.guidance.pid_core import clamp
from src.python.navigation.nav_integration import NavigationSystem
from src.python.vision.integrated_vision import IntegratedVisionSystem


def load_config():
    config_path = PROJECT_ROOT / "config" / "pid_params.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── Ana Program ──────────────────────────────────────────────────────────────
def main():
    config = load_config()

    # 1. Vision ve Dosya Yapılandırma Ayarlarını Yükle
    vision_cfg = config.get("vision", {})
    resize_scale = float(vision_cfg.get("resize_scale", 0.7))
    monitor_index = int(vision_cfg.get("monitor_index", 1))
    output_fps = float(vision_cfg.get("output_fps", 50.0))
    
    output_dir = Path("/home/tom/Documents/Projeler/DogFight")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sim_gorsel_takip_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

    # 2. Entegre Görsel Takip Sistemini Yükle
    vision_system = IntegratedVisionSystem(config)

    # 3. Simülasyon Kontrolcülerini Yükle
    mode_mgr = ModeManager(config)
    pitch_ctrl = PitchController(config)
    pitch_ctrl.camera_tilt = 0.0  # Kamera tilt açısı kaldırıldı (sabit ilerleme sorunu için)
    throttle_ctrl = ThrottleController(config)
    yaw_ctrl = YawController(config)
    nav = NavigationSystem(corrupt_for_test=False)

    # 4. Oyuna Bağlan
    connected = drone.connect()
    if not connected or not drone.is_connected():
        print("[ERROR] Oyuna bağlanılamadı. DronesOfWar simülasyonunun açık olduğundan emin olun.")
        return

    drone.set_arm(True)
    print("[INFO] Entegre Görsel Kılavuzluk Sistemi Hazır. Simülasyon başlatılıyor...")

    # Ekran Yakalama Başlat
    video_writer = None
    video_time_accumulator = 0.0
    prev_time = time.perf_counter()
    fps = 0.0
    fps_alpha = 0.9

    loop_time = time.perf_counter()

    with mss.MSS() as sct:
        if monitor_index >= len(sct.monitors):
            raise RuntimeError(f"Geçersiz monitör indeksi: {monitor_index}")

        monitor = sct.monitors[monitor_index]
        print(f"[INFO] Ekran Yakalanıyor: {monitor}")
        print(f"[INFO] Video Kayıt Yolu: {output_path}")

        try:
            while True:
                # Zaman ve döngü hesapları
                now = time.perf_counter()
                dt = now - loop_time
                loop_time = now
                dt = max(0.01, min(0.1, dt))

                # Görüntü yakala
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                if resize_scale != 1.0:
                    frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)

                # FPS hesabı
                dt_fps = now - prev_time
                if dt_fps > 0:
                    cur_fps = 1.0 / dt_fps
                    fps = (fps_alpha * fps + (1.0 - fps_alpha) * cur_fps) if fps > 0 else cur_fps
                prev_time = now

                # Telemetri oku
                tel = drone.get_telemetry()
                drone_pos = tel["drone"]["position"]
                drone_rot = tel["drone"]["rotation"]
                drone_speed = tel["drone"]["speed"]
                raw_target_pos = tel["target"]["position"]

                # Telemetri yoksa bekle
                if raw_target_pos[0] == 0.0 and drone_pos[0] == 0.0:
                    time.sleep(0.01)
                    continue

                # EKF & Konum Hesaplama
                nav_out = nav.process(tel)
                dx = nav_out["dx"]
                dy = nav_out["dy"]
                dz = nav_out["dz"]
                dist_2d = nav_out["dist_2d"]
                dist_3d = nav_out["dist_3d"]
                target_speed = nav_out["target_speed"]
                target_elevation = math.degrees(math.atan2(dz, max(dist_2d, 1.0)))
                gps_ok = nav_out["gps_ok"]

                # ─── GÖRSEL TESPİT VE TAKİP DURUM MAKİNESİ ───────────────────
                vis_res = vision_system.update(frame, dist_3d, dz, dt)
                tracking = vis_res["tracking"]
                display_frame = vis_res["display_frame"]

                # ─── KILAVUZLUK VE KONTROL MODU SEÇİMİ ─────────────────────────
                if tracking:
                    # MOD 1: KAMERA TABANLI GÖRSEL KILAVUZLUK (VISION GUIDANCE)
                    yaw_cmd = vis_res["yaw_cmd"]
                    pitch_cmd = vis_res["pitch_cmd"]
                    yaw_error = vis_res["yaw_err"]
                    
                    target_throttle = throttle_ctrl.calculate(dz=dz, dt=dt)
                    roll_cmd = 0.0
                    guide_mode = "VISION"
                else:
                    # MOD 2: EKF/GPS TABANLI KONUM KILAVUZLUĞU (POSITION GUIDANCE)
                    guide_mode = "POSITION"
                    if not gps_ok or not nav.target_ekf.initialized:
                        target_throttle = clamp(config["flight"]["base_throttle"] + 0.15, -1.0, 1.0)
                        pitch_cmd = 0.0
                        yaw_cmd = 0.0
                        roll_cmd = 0.0
                        yaw_error = 0.0
                    else:
                        yaw_error, yaw_cmd = yaw_ctrl.calculate(dx, dy, drone_rot[2], dt)
                        state, target_speed_cmd = mode_mgr.update(dist_3d, dz, drone_speed, target_speed, yaw_error, dt)
                        _, pitch_cmd = pitch_ctrl.calculate(state, target_speed_cmd, drone_speed, target_elevation, drone_rot[1], yaw_error, dt)
                        target_throttle = throttle_ctrl.calculate(dz=dz, dt=dt)
                        roll_cmd = 0.0

                # ─── YÜKSEKLİK VE YUNUSLAMA SINIRLANDIRMASI (HEDEFTEN YÜKSEK OLMA DURUMU) ───
                if dz < 0.0:
                    # Drone hedeften daha yüksekteyse (dz < 0), throttle'ın base_throttle'ı aşmasını engelle
                    target_throttle = min(target_throttle, throttle_ctrl.base_throttle)
                    # Yunuslama yukarı (pitch_cmd > 0) komutunu sınırla
                    max_allowed_pitch_up = max(0.0, 1.0 + (dz / 200.0))
                    if pitch_cmd > max_allowed_pitch_up:
                        pitch_cmd = max_allowed_pitch_up

                # ─── KONTROL YÜZEYLERİNİ GÜNCELLE ──────────────────────────────
                drone.set_control_surfaces(
                    clamp(target_throttle, -1.0, 1.0),
                    -clamp(pitch_cmd, -1.0, 1.0),
                    -clamp(roll_cmd, -1.0, 1.0),
                    clamp(yaw_cmd, -1.0, 1.0),
                    True,
                )

                # HUD ekranını çiz
                vision_system.draw_hud(
                    display_frame,
                    mode=guide_mode,
                    remaining_time=vis_res["remaining_time"],
                    target_class=vis_res["target_class"],
                    tracker_score=vis_res["tracker_score"],
                    fps=fps,
                    dist=dist_3d,
                    dz=dz,
                    yaw_err=yaw_error,
                    pitch_cmd=pitch_cmd,
                    yaw_cmd=yaw_cmd,
                    throttle=target_throttle,
                    detection_score=vis_res["detection_score"]
                )

                # Video Kaydı (Gerçek zamanlı hız eşleme için kare biriktirici)
                if video_writer is None:
                    oh, ow = display_frame.shape[:2]
                    video_writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (ow, oh),
                    )
                
                # dt süresini biriktirerek videoyu gerçek zamanlı hıza eşitler
                video_time_accumulator += dt
                frame_duration = 1.0 / output_fps
                while video_time_accumulator >= frame_duration:
                    video_writer.write(display_frame)
                    video_time_accumulator -= frame_duration

                # Pencere Gösterimi
                cv2.imshow("RF-DETR + TCTrack + SimBridge Entegre Gorsek Kilavuzluk", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("t"):
                    vision_system.tracker_enabled = not vision_system.tracker_enabled
                    # Mod değiştirildiğinde aktif takibi temizle
                    vision_system.tracking = False
                    vision_system.tracked_bbox = None
                    print(f"[INFO] Takip modu değiştirildi: {'TCTRACK' if vision_system.tracker_enabled else 'SADECE TESPİT (DETECT-ONLY)'}")

        except KeyboardInterrupt:
            print("\n[INFO] Kullanıcı tarafından sonlandırıldı.")
        finally:
            # Güvenli Kapatma
            drone.set_throttle(0.0)
            drone.set_arm(False)
            drone.disconnect()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            print(f"[INFO] Video Kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
