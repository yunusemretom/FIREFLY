"""
Bridge connecting Python backend to Gazebo/ROS2 simulator environments.
"""

# src/python/simulation/sim_bridge.py

import time
import math
import yaml
import sys
import os

# ruff: noqa: E402
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import drone_sdk as drone
from src.python.guidance.mode_manager import ModeManager
from src.python.guidance.pid_pitch import PitchController
from src.python.guidance.pid_throttle import ThrottleController
from src.python.guidance.pid_yaw import YawController
from src.python.guidance.pid_core import clamp


def load_config():
    with open("config/pid_params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_simulation():
    config = load_config()

    mode_mgr = ModeManager(config)
    pitch_ctrl = PitchController(config)
    throttle_ctrl = ThrottleController(config)
    yaw_ctrl = YawController(config)

    drone.connect()
    time.sleep(0.5)
    drone.set_arm(True)

    print("=" * 60)
    print("  AVCI DRONE - TAKİP SİMÜLASYONU BAŞLADI")
    print("=" * 60)

    loop_time = time.time()

    try:
        while True:
            # Measure real elapsed time instead of using fixed dt
            now = time.time()
            dt = now - loop_time
            loop_time = now
            dt = max(0.01, min(0.1, dt))

            # A) Telemetriyi Oku
            tel = drone.get_telemetry()
            drone_pos = tel["drone"]["position"]
            drone_rot = tel["drone"]["rotation"]
            drone_speed = tel["drone"]["speed"]
            target_pos = tel["target"]["position"]
            target_speed = tel["target"]["speed"]

            if target_pos[0] == 0.0 and drone_pos[0] == 0.0:
                time.sleep(0.02)
                continue

            # B) Geometriyi Hesapla
            dx = target_pos[0] - drone_pos[0]
            dy = target_pos[1] - drone_pos[1]
            dz = target_pos[2] - drone_pos[2]

            dist_2d = math.sqrt(dx**2 + dy**2)
            dist_3d = math.sqrt(dx**2 + dy**2 + dz**2)
            target_elevation = math.degrees(math.atan2(dz, max(dist_2d, 1.0)))

            # C) Kontrol Kararları

            # 1. Yaw: hedefe dön
            yaw_error, yaw_cmd = yaw_ctrl.calculate(dx, dy, drone_rot[2], dt)

            # 2. Durum güncelle
            state, target_speed_cmd = mode_mgr.update(
                dist_3d=dist_3d,
                dz=dz,
                drone_speed=drone_speed,
                target_speed=target_speed,
                yaw_error=yaw_error,
                dt=dt,
            )

            # 3. Pitch: hedefe dönüklük oranında ileri eğim uygula (yaw hizasıyla scale edilir)
            smooth_desired_pitch, pitch_cmd = pitch_ctrl.calculate(
                state=state,
                target_speed_cmd=target_speed_cmd,
                drone_speed=drone_speed,
                target_elevation=target_elevation,
                drone_pitch=drone_rot[1],
                yaw_error=yaw_error,
                dt=dt,
            )

            # 4. Throttle: irtifa koru (0=hover, SDK [-1,1])
            target_throttle = throttle_ctrl.calculate(dz=dz, dt=dt)

            # 5. Roll: angle mode FC seviyede tutar, 0 gönder
            roll_cmd = 0.0

            # D) Komut Gönder
            drone.set_control_surfaces(
                clamp(target_throttle, -1.0, 1.0),
                -clamp(pitch_cmd, -1.0, 1.0),
                -roll_cmd,
                clamp(yaw_cmd, -1.0, 1.0),
                True,
            )

            print(
                f"[{state}] dist={dist_3d:.0f} dz={dz:.0f} hız={drone_speed / 27.78:.1f}km/h | "
                f"yaw_err={yaw_error:.1f}° pit={smooth_desired_pitch:.1f}° thr={target_throttle:.2f}"
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
