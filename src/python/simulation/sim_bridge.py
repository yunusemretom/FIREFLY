
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

from src.python.navigation.nav_integration import NavigationSystem


def load_config():
    with open("config/pid_params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_simulation():
    config = load_config()

    mode_mgr = ModeManager(config)
    pitch_ctrl = PitchController(config)
    throttle_ctrl = ThrottleController(config)
    yaw_ctrl = YawController(config)

    nav = NavigationSystem(corrupt_for_test=False)

    connected = drone.connect()
    print(f"[TANI] drone.connect() sonucu: {connected}")
    print(f"[TANI] drone.is_connected(): {drone.is_connected()}")
    if not connected or not drone.is_connected():
        print(
            "\n[!] OYUNA BAGLANILAMADI. Kontrol et:\n"
            "    1) DronesOfWar.exe (oyun) acik mi?\n"
            "    2) BASKA BIR SCRIPT (ornegin drone_gui.py) su an ayni porta "
            "(127.0.0.1:12345) BAGLI DEGIL mi? Ayni anda ikisi calisamaz - "
            "birini kapat.\n"
        )
        return
    time.sleep(0.5)

    print("[TANI] Ilk telemetri ornekleri (1.5s):")
    _t0 = time.time()
    while time.time() - _t0 < 1.5:
        _tel = drone.get_telemetry()
        print(f"    drone_pos={_tel['drone']['position']}  target_pos={_tel['target']['position']}")
        time.sleep(0.3)

    drone.set_arm(True)

    print("=" * 60)
    print("  AVCI DRONE - TAKİP SİMÜLASYONU BAŞLADI (EKF entegre)")
    print("=" * 60)

    loop_time = time.time()

    try:
        while True:
            now = time.time()
            dt = now - loop_time
            loop_time = now
            dt = max(0.01, min(0.1, dt))

            tel = drone.get_telemetry()
            drone_pos = tel["drone"]["position"]
            drone_rot = tel["drone"]["rotation"]
            drone_speed = tel["drone"]["speed"]
            raw_target_pos = tel["target"]["position"]

            if raw_target_pos[0] == 0.0 and drone_pos[0] == 0.0:
                time.sleep(0.02)
                continue


            if raw_target_pos == [0.0, 0.0, 0.0] or tuple(raw_target_pos) == (0.0, 0.0, 0.0):
                print("  (!) Hedef telemetrisi (0,0,0) - oyun/GPS baglantisini kontrol et")
                time.sleep(0.1)
                continue

            nav_out = nav.process(tel)

            dx = nav_out["dx"]
            dy = nav_out["dy"]
            dz = nav_out["dz"]
            dist_2d = nav_out["dist_2d"]
            dist_3d = nav_out["dist_3d"]
            target_speed = nav_out["target_speed"]
            target_elevation = math.degrees(math.atan2(dz, max(dist_2d, 1.0)))

            gps_ok = nav_out["gps_ok"]


            if not gps_ok or not nav.target_ekf.initialized:
                state = "KALKIS_BEKLEME"
                target_throttle = clamp(config["flight"]["base_throttle"] + 0.15, -1.0, 1.0)  # yavas, guvenli tirmanma
                pitch_cmd = 0.0
                yaw_cmd = 0.0
                roll_cmd = 0.0
                smooth_desired_pitch = 0.0
                yaw_error = 0.0

                drone.set_control_surfaces(
                    clamp(target_throttle, -1.0, 1.0), 0.0, 0.0, 0.0, True,
                )
                print(
                    f"[{state}][!GPS!] EKF isinmadi, guvenli hover komutu gonderiliyor "
                    f"(target_ekf.initialized={nav.target_ekf.initialized})"
                )
                time.sleep(max(0.0, config["flight"]["dt"] - dt))
                continue


            yaw_error, yaw_cmd = yaw_ctrl.calculate(dx, dy, drone_rot[2], dt)

            state, target_speed_cmd = mode_mgr.update(
                dist_3d=dist_3d,
                dz=dz,
                drone_speed=drone_speed,
                target_speed=target_speed,
                yaw_error=yaw_error,
                dt=dt,
            )

            smooth_desired_pitch, pitch_cmd = pitch_ctrl.calculate(
                state=state,
                target_speed_cmd=target_speed_cmd,
                drone_speed=drone_speed,
                target_elevation=target_elevation,
                drone_pitch=drone_rot[1],
                yaw_error=yaw_error,
                dt=dt,
            )

            target_throttle = throttle_ctrl.calculate(dz=dz, dt=dt)

            roll_cmd = 0.0

            drone.set_control_surfaces(
                clamp(target_throttle, -1.0, 1.0),
                -clamp(pitch_cmd, -1.0, 1.0),
                -clamp(roll_cmd, -1.0, 1.0),
                clamp(yaw_cmd, -1.0, 1.0),
                True,
            )

            gps_flag = "OK" if gps_ok else "!GPS!"
            _active_corr = " | ".join(drone.get_active_corruption())
            print(
                f"[{state}][{gps_flag}] dist={dist_3d:.0f} dz={dz:.0f} hız={drone_speed / 27.78:.1f}km/h | "
                f"yaw_err={yaw_error:.1f}° pit={smooth_desired_pitch:.1f}° thr={target_throttle:.2f}"
                + (f"  || BOZMA: {_active_corr}" if _active_corr else "")
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