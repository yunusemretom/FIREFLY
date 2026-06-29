"""
sim_bridge_pursuit.py — 5 metre arkadan takip köprüsü (test versiyonu)

Orijinal sim_bridge.py'ye dokunulmaz.

Güdüm mantığı:
  - Hedefin hız vektörünün tersine follow_distance_cm (varsayılan 500 cm = 5 m)
    uzaklıkta bir "takip noktası" hesaplanır.
  - Drone bu noktayı hedef alır — hedefin kendisini değil.
  - İmzalı mesafe s = (drone − takip_noktası) · hedef_yönü:
      s > 0 → drone çok yakın / geçme tehlikesi → hız düşürülür
      s = 0 → doğru konumda              → hedef hızıyla eşleşilir
      s < 0 → geri kaldı                → hızlanılır
  - Throttle + Pitch eş zamanlı çalışır (KALKIŞ bekleme yok).
  - follow_distance_cm config/pid_params.yaml → pursuit → follow_distance_cm

Çalıştırmak için:
    python3 tests/python/sim_bridge_pursuit.py
"""

import time
import math
import yaml
import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import drone_sdk as drone
from src.python.guidance.pid_core import clamp
from src.python.guidance.pursuit import PursuitController
from src.python.navigation.ekf import TargetEKF


def load_config():
    with open("config/pid_params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _best_target_pos(raw_pos, truth_data, ekf, now):
    """Truth > EKF > ham GPS öncelik sırası. EKF her durumda beslenir."""
    if truth_data["available"]:
        truth_pos = truth_data["target"]["position"]
        ekf.update(truth_pos, timestamp=now)
        return truth_pos, "TRUTH"
    filtered = ekf.update(raw_pos, timestamp=now)
    if not ekf.is_ready:
        return raw_pos, "RAW"
    tag = "EKF"
    if ekf.last_spike_rejected:
        tag = "EKF[SPIKE]"
    elif ekf.last_dropout:
        tag = "EKF[DROPOUT]"
    return filtered, tag


def run_simulation():
    config  = load_config()
    ekf     = TargetEKF()
    pursuit = PursuitController(config)

    follow_dist = config.get("pursuit", {}).get("follow_distance_cm", 500.0)

    drone.connect()
    time.sleep(0.5)
    drone.set_arm(True)

    print("=" * 70)
    print(f"  AVCI DRONE — PURSUIT TAKİP  [takip mesafesi = {follow_dist:.0f} cm = {follow_dist/100:.1f} m]")
    print("=" * 70)
    print(
        f"  {'Kaynak':<14} {'s(cm)':>8} {'HedefHz':>8} {'DroneHz':>8} "
        f"{'yaw°':>6} {'pit°':>6} {'thr':>5}"
    )
    print("  " + "-" * 63)

    loop_time = time.time()

    try:
        while True:
            now       = time.time()
            dt        = now - loop_time
            loop_time = now
            dt        = max(0.01, min(0.1, dt))

            # A) Telemetri
            tel          = drone.get_telemetry()
            drone_pos    = tel["drone"]["position"]
            drone_rot    = tel["drone"]["rotation"]
            drone_speed  = tel["drone"]["speed"]
            raw_pos      = tel["target"]["position"]

            if raw_pos[0] == 0.0 and drone_pos[0] == 0.0:
                time.sleep(0.02)
                continue

            # B) En iyi hedef konumu (EKF her durumda beslenir)
            truth_data          = drone.get_debug_truth()
            target_pos, src_tag = _best_target_pos(raw_pos, truth_data, ekf, now)

            # C) EKF'den hız vektörü (takip noktası hesabı için)
            target_vel = ekf.get_estimated_velocity() if ekf.is_ready else (1500.0, 0.0, 0.0)

            # D) Güdüm hesapla
            commands, telem = pursuit.calculate(
                drone_pos  = drone_pos,
                drone_rot  = drone_rot,
                drone_speed= drone_speed,
                target_pos = target_pos,
                target_vel = target_vel,
                dt         = dt,
            )
            throttle, pitch_cmd, roll_cmd, yaw_cmd = commands

            # E) Komut gönder
            drone.set_control_surfaces(
                clamp(throttle,  -1.0, 1.0),
                -clamp(pitch_cmd, -1.0, 1.0),
                -clamp(roll_cmd,  -1.0, 1.0),
                clamp(yaw_cmd,   -1.0, 1.0),
                True,
            )

            # F) Terminal çıktısı
            s    = telem["signed_dist_cm"]
            vt   = telem["target_speed"] / 27.78    # km/h
            vd   = drone_speed / 27.78
            yaw  = telem["yaw_error"]
            pit  = telem["smooth_pitch"]
            thr  = throttle

            # s işaret göstergesi: negatif = geri, pozitif = çok yakın/geçti
            s_indicator = "YAKIN" if s > follow_dist * 0.5 else ("GERİ" if s < -100 else "OK")

            print(
                f"  {src_tag:<14} {s:>+8.0f} {vt:>7.1f}k {vd:>7.1f}k "
                f"{yaw:>+6.1f} {pit:>+6.1f} {thr:>+5.2f}  {s_indicator}",
                end="\r",
            )

            time.sleep(max(0.0, config["flight"]["dt"] - dt))

    except KeyboardInterrupt:
        print("\n\n[!] Kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n\n[!] Kritik Hata: {e}")
        import traceback; traceback.print_exc()
    finally:
        drone.set_throttle(0.0)
        drone.set_arm(False)
        drone.disconnect()


if __name__ == "__main__":
    run_simulation()
