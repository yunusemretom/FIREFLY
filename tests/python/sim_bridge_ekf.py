"""
sim_bridge_ekf.py — Agresif takip köprüsü (test versiyonu)

Orijinal sim_bridge.py'ye dokunulmaz.  Bu dosyadaki üç temel fark:

  1. Hedef konumu: truth (debug Hz) > EKF > ham GPS öncelik sırasıyla.
  2. KALKIŞ bekleme yok: throttle (irtifa) ve pitch (ilerleme) her zaman eş zamanlı çalışır.
  3. Agresif mesafe kapama: yaw_scale min 0.3 — dönüş sırasında da ileri gidilir;
     hedef hızından bağımsız olarak mümkün olan maksimum hızla yaklaşılır.
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
from src.python.guidance.pid_core import PID, SlewLimiter, clamp
from src.python.guidance.pid_yaw import YawController
from src.python.guidance.pid_throttle import ThrottleController
from src.python.navigation.ekf import TargetEKF


def load_config():
    with open("config/pid_params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _best_target_pos(raw_pos, truth_data, ekf, now):
    """Truth > EKF > ham GPS öncelik sırası."""
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
    config = load_config()

    # Throttle ve Yaw: orijinal kontrolörler yeterli
    throttle_ctrl = ThrottleController(config)
    yaw_ctrl      = YawController(config)
    ekf           = TargetEKF()

    # Pitch: doğrudan PID + slew — KALKIŞ bloğu ve sert yaw_scale yok
    sp = config["outer_pid"]["speed"]
    speed_pid = PID(
        kp=sp["kp"], ki=sp["ki"], kd=sp["kd"],
        clamp=(sp["clamp_min"], sp["clamp_max"]),
        integral_limit=sp["int_limit"],
    )
    ip = config["inner_pid"]["pitch"]
    pitch_inner = PID(kp=ip["kp"], ki=ip["ki"], kd=ip["kd"], clamp=(-1.0, 1.0))
    slew          = SlewLimiter(config["slew"]["pitch_max_change_per_sec"])

    camera_tilt  = config["flight"]["camera_tilt"]   # derece
    min_pitch    = config["limits"]["min_pitch"]      # negatif (ör. -60)
    max_pitch    = config["limits"]["max_pitch"]      # pozitif (ör. 20)
    chase_speed  = config["flight"]["chase_speed"]    # cm/s (ör. 2800)
    takip_dist   = config["thresholds"]["takip_dist"] # cm (ör. 8000)
    takip_margin = config["flight"]["takip_speed_margin"]  # cm/s (ör. 500)

    drone.connect()
    time.sleep(0.5)
    drone.set_arm(True)

    print("=" * 65)
    print("  AVCI DRONE — AGRESIF TAKİP  [TRUTH > EKF | EŞ ZAMANLI]")
    print("=" * 65)

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

            # B) En iyi hedef konumu
            truth_data          = drone.get_debug_truth()
            target_pos, src_tag = _best_target_pos(raw_pos, truth_data, ekf, now)

            # C) Geometri
            dx = target_pos[0] - drone_pos[0]
            dy = target_pos[1] - drone_pos[1]
            dz = target_pos[2] - drone_pos[2]

            dist_2d          = math.sqrt(dx**2 + dy**2)
            dist_3d          = math.sqrt(dx**2 + dy**2 + dz**2)
            target_elevation = math.degrees(math.atan2(dz, max(dist_2d, 1.0)))

            # D) Yaw: hedefe dön (değişmedi)
            yaw_error, yaw_cmd = yaw_ctrl.calculate(dx, dy, drone_rot[2], dt)

            # E) Throttle: irtifa (değişmedi, her zaman aktif)
            target_throttle = throttle_ctrl.calculate(dz=dz, dt=dt)

            # F) Pitch: KALKIŞ yok, eş zamanlı, agresif
            #
            # Hız hedefi:
            #   - Uzaktaysa (> takip_dist): tam kovalama hızı — mesafeyi kapat
            #   - Yakındaysa              : hedef hızı + marj — çarpışmadan yakın kal
            if dist_3d > takip_dist:
                target_speed_cmd = chase_speed
                state_label = "SEYIR"
            else:
                target_speed_cmd = 1500.0 + takip_margin
                state_label = "TAKIP"

            speed_error_ms  = (target_speed_cmd - drone_speed) / 100.0
            speed_adj       = speed_pid.calculate(speed_error_ms, dt)

            # Kamera açısı hedefe bakacak şekilde pitch belirler;
            # hız hatası düzeltmesi bunu yukarı/aşağı iter.
            camera_center   = target_elevation - camera_tilt
            raw_desired     = clamp(camera_center - speed_adj, min_pitch, max_pitch)
            smooth_pitch    = slew.update(raw_desired, dt)

            # [-1, 1] aralığına normalize
            if smooth_pitch < 0:
                pitch_cmd = smooth_pitch / abs(min_pitch) if min_pitch != 0 else 0.0
            else:
                pitch_cmd = smooth_pitch / abs(max_pitch) if max_pitch != 0 else 0.0

            # Yaw hizalaması ölçekleme — min 0.3 ile dönüş sırasında da ileri gidilir
            # (orijinalde max(0, cos) → 90°'de tamamen sıfır; burası max(0.3, cos))
            yaw_scale = max(0.3, math.cos(math.radians(yaw_error)))
            pitch_cmd = clamp(pitch_cmd * yaw_scale, -1.0, 1.0)

            roll_cmd = 0.0

            # G) Komut gönder
            drone.set_control_surfaces(
                clamp(target_throttle, -1.0, 1.0),
                -clamp(pitch_cmd, -1.0, 1.0),
                -clamp(roll_cmd, -1.0, 1.0),
                clamp(yaw_cmd, -1.0, 1.0),
                True,
            )

            print(
                f"[{state_label}|{src_tag}] "
                f"dist={dist_3d:6.0f} dz={dz:+5.0f} "
                f"hız={drone_speed/27.78:5.1f}km/h "
                f"hedef={target_speed_cmd/27.78:.0f}km/h | "
                f"yaw={yaw_error:+5.1f}° pit={smooth_pitch:+5.1f}° "
                f"thr={target_throttle:+.2f}"
            )

            time.sleep(max(0.0, config["flight"]["dt"] - dt))

    except KeyboardInterrupt:
        print("\n[!] Kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n[!] Kritik Hata: {e}")
    finally:
        drone.set_throttle(0.0)
        drone.set_arm(False)
        drone.disconnect()


if __name__ == "__main__":
    run_simulation()
