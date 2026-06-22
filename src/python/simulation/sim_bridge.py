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
from src.python.guidance.pid_core import PID, clamp


def load_config():
    """YAML ayar dosyasını okur ve sözlük (dict) olarak döner."""
    with open("config/pid_params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_simulation():
    config = load_config()
    dt = config["flight"]["dt"]

    # 1. Aşama: Otonom Zekayı (Modülleri) Başlat
    mode_mgr = ModeManager(config)
    pitch_ctrl = PitchController(config)
    throttle_ctrl = ThrottleController(config)
    yaw_ctrl = YawController(config)

    # Not: Ayrı bir pid_roll.py oluşturmadıysak ufuk çizgisini (roll) düzeltmek için PID'yi burada anlık başlatıyoruz
    ip_roll = config["inner_pid"]["roll"]
    roll_pid = PID(
        kp=ip_roll["kp"], ki=ip_roll["ki"], kd=ip_roll["kd"], clamp=(-1.0, 1.0)
    )

    # 2. Aşama: Simülasyona Bağlan
    drone.connect()
    time.sleep(0.5)
    drone.set_arm(True)

    print("=" * 60)
    print("  AVCI DRONE - GÜDÜMLÜ FÜZE (MODÜLER) SİMÜLASYONU BAŞLADI")
    print("=" * 60)

    try:
        while True:
            # A) Telemetriyi Oku
            tel = drone.get_telemetry()
            drone_pos = tel["drone"]["position"]
            drone_rot = tel["drone"]["rotation"]
            drone_speed = tel["drone"]["speed"]
            target_pos = tel["target"]["position"]
            target_speed = tel["target"]["speed"]

            if target_pos[0] == 0.0 and drone_pos[0] == 0.0:
                time.sleep(0.1)
                continue

            # B) Fiziksel Mesafeleri (Geometriyi) Hesapla
            dx = target_pos[0] - drone_pos[0]
            dy = target_pos[1] - drone_pos[1]
            dz = target_pos[2] - drone_pos[2]

            dist_2d = math.sqrt(dx**2 + dy**2)
            dist_3d = math.sqrt(dx**2 + dy**2 + dz**2)
            target_elevation = math.degrees(math.atan2(dz, max(dist_2d, 1.0)))

            # C) YAPAY ZEKA KARARLARI (Modüllere Sor)

            # 1. Hangi yöne dönmeliyiz? (Yaw)
            yaw_error, yaw_cmd = yaw_ctrl.calculate(dx, dy, drone_rot[2], dt)

            # 2. Hangi durumdayız (Seyir mi, Takip mi, Angajman mı) ve hızımız ne olmalı?
            state, target_speed_cmd = mode_mgr.update(
                dist_3d=dist_3d,
                dz=dz,
                drone_speed=drone_speed,
                target_speed=target_speed,
                yaw_error=yaw_error,
                dt=dt,
            )

            # 3. İstenilen hıza ulaşmak için burnumuzu ne kadar ezmeliyiz? (Pitch)
            smooth_desired_pitch, pitch_cmd = pitch_ctrl.calculate(
                state=state,
                target_speed_cmd=target_speed_cmd,
                drone_speed=drone_speed,
                target_elevation=target_elevation,
                drone_pitch=drone_rot[1],
                dt=dt,
            )

            # 4. Burnumuzu ezdiğimiz (veya kaldırdığımız) açıda irtifayı korumak için motorlara ne kadar güç vermeliyiz? (Throttle)
            target_throttle = throttle_ctrl.calculate(
                dz=dz, current_pitch=smooth_desired_pitch, dt=dt
            )

            # 5. Rüzgar vb. bizi sağa sola yatırırsa yatay (Ufuk) pozisyonuna dön. (Roll)
            roll_error = 0.0 - drone_rot[0]
            roll_cmd = clamp(roll_pid.calculate(roll_error, dt), -1.0, 1.0)

            # D) KOMUTLARI GÖNDER VE LOGLA

            # Güvenlik (Çıkışları SDK için sınırla)
            pitch_cmd = clamp(pitch_cmd, -1.0, 1.0)
            yaw_cmd = clamp(yaw_cmd, -1.0, 1.0)

            drone.set_control_surfaces(
                target_throttle, pitch_cmd, roll_cmd, yaw_cmd, True
            )

            # Konsol Çıktısı
            print(
                f"[{state}] dist={dist_3d:.0f} dz={dz:.0f} hız={drone_speed / 27.78:.1f}km/h | "
                f"D_Pit={smooth_desired_pitch:.1f}° A_Pit={drone_rot[1]:.1f}° T={target_throttle:.2f}"
            )

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[!] Kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n[!] Kritik Hata: {e}")
    finally:
        # Program kapanırken dronu yere çakılmaması için güvenli moda al
        drone.set_throttle(0.0)
        drone.set_arm(False)
        drone.disconnect()


if __name__ == "__main__":
    run_simulation()
