
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


def update_plots(fig, axes, t, d_pos, t_pos_raw, t_pos_ekf, d_spd, t_spd, dist_3d, y_err, p_cmd, y_cmd, thr_cmd, final=False):
    import numpy as np
    
    ax_xy, ax_z, ax_speed_dist, ax_control = axes.flat
    
    d_pos = np.array(d_pos)
    t_pos_raw = np.array(t_pos_raw)
    t_pos_ekf = np.array(t_pos_ekf)
    t = np.array(t)
    
    # 1. XY Trajectory Plot (Drone vs Target Raw vs Target EKF)
    ax_xy.clear()
    ax_xy.set_title("XY Düzlemi Yörüngesi (cm)")
    ax_xy.set_xlabel("X (cm)")
    ax_xy.set_ylabel("Y (cm)")
    
    if len(d_pos) > 0:
        ax_xy.plot(d_pos[:, 0], d_pos[:, 1], color='#00e5ff', label='Avcı Drone', linewidth=2)
        ax_xy.scatter(d_pos[0, 0], d_pos[0, 1], color='green', marker='o', s=50, label='Başlangıç (Drone)')
        ax_xy.scatter(d_pos[-1, 0], d_pos[-1, 1], color='red', marker='x', s=50)
        
    if len(t_pos_raw) > 0:
        ax_xy.scatter(t_pos_raw[:, 0], t_pos_raw[:, 1], color='#ff3d00', s=4, alpha=0.3, label='Ham Hedef GPS')
        
    if len(t_pos_ekf) > 0:
        ax_xy.plot(t_pos_ekf[:, 0], t_pos_ekf[:, 1], color='#00e676', label='Hedef EKF Tahmini', linewidth=1.5)
        ax_xy.scatter(t_pos_ekf[0, 0], t_pos_ekf[0, 1], color='lightgreen', marker='o', s=30)
        ax_xy.scatter(t_pos_ekf[-1, 0], t_pos_ekf[-1, 1], color='darkgreen', marker='x', s=40)
        
    ax_xy.legend(loc='best', facecolor='#2d2d2d', edgecolor='#444444', labelcolor='white', fontsize=8)
    ax_xy.set_aspect('equal', 'datalim')
    
    # 2. Z Altitude Profile over Time
    ax_z.clear()
    ax_z.set_title("İrtifa Değişimi (Z ekseni)")
    ax_z.set_xlabel("Zaman (s)")
    ax_z.set_ylabel("Yükseklik (cm)")
    
    if len(d_pos) > 0:
        ax_z.plot(t, d_pos[:, 2], color='#00e5ff', label='Avcı Drone', linewidth=2)
    if len(t_pos_raw) > 0:
        ax_z.plot(t, t_pos_raw[:, 2], color='#ff3d00', alpha=0.3, label='Ham Hedef GPS', linestyle='None', marker='.', markersize=2)
    if len(t_pos_ekf) > 0:
        ax_z.plot(t, t_pos_ekf[:, 2], color='#00e676', label='Hedef EKF', linewidth=1.5)
        
    ax_z.legend(loc='best', facecolor='#2d2d2d', edgecolor='#444444', labelcolor='white', fontsize=8)
    
    # 3. Distance & Speed Plot
    ax_speed_dist.clear()
    ax_speed_dist.set_title("Mesafe ve Hız Grafiği")
    ax_speed_dist.set_xlabel("Zaman (s)")
    ax_speed_dist.set_ylabel("Mesafe (cm)", color='#e0e0e0')
    
    line_dist = ax_speed_dist.plot(t, dist_3d, color='#ffeb3b', label='3D Mesafe', linewidth=2)
    ax_speed_dist.tick_params(axis='y', labelcolor='#ffeb3b')
    
    ax_speed = ax_speed_dist.twinx()
    ax_speed.set_ylabel("Hız (km/h)", color='#e0e0e0')
    
    d_spd_kmh = np.array(d_spd) * 0.036
    t_spd_kmh = np.array(t_spd) * 0.036
    
    line_d_spd = ax_speed.plot(t, d_spd_kmh, color='#00e5ff', label='Drone Hızı', linewidth=1.2, linestyle='--')
    line_t_spd = ax_speed.plot(t, t_spd_kmh, color='#00e676', label='Hedef Hızı', linewidth=1.2, linestyle=':')
    ax_speed.tick_params(axis='y', labelcolor='#00e5ff')
    ax_speed.grid(False)
    
    lines = line_dist + line_d_spd + line_t_spd
    labels = [l.get_label() for l in lines]
    ax_speed_dist.legend(lines, labels, loc='best', facecolor='#2d2d2d', edgecolor='#444444', labelcolor='white', fontsize=8)
    
    # 4. Control Commands & Yaw Error
    ax_control.clear()
    ax_control.set_title("Kontrol Komutları ve Yaw Hatası")
    ax_control.set_xlabel("Zaman (s)")
    ax_control.set_ylabel("Yaw Hatası (Derece)", color='#ff5722')
    
    line_yaw_err = ax_control.plot(t, y_err, color='#ff5722', label='Yaw Hatası (°)', linewidth=1.5)
    ax_control.tick_params(axis='y', labelcolor='#ff5722')
    
    ax_cmd = ax_control.twinx()
    ax_cmd.set_ylabel("Komut Değeri (Normalizasyon)", color='#e0e0e0')
    
    line_pitch = ax_cmd.plot(t, p_cmd, color='#e91e63', label='Pitch Komutu', linewidth=1, linestyle='-')
    line_yaw_cmd = ax_cmd.plot(t, y_cmd, color='#9c27b0', label='Yaw Komutu', linewidth=1, linestyle='--')
    line_thr = ax_cmd.plot(t, thr_cmd, color='#009688', label='Throttle Komutu', linewidth=1, linestyle='-.')
    
    ax_cmd.set_ylim(-1.1, 1.1)
    ax_cmd.tick_params(axis='y', labelcolor='#e0e0e0')
    ax_cmd.grid(False)
    
    lines_cmd = line_yaw_err + line_pitch + line_yaw_cmd + line_thr
    labels_cmd = [l.get_label() for l in lines_cmd]
    ax_control.legend(lines_cmd, labels_cmd, loc='best', facecolor='#2d2d2d', edgecolor='#444444', labelcolor='white', fontsize=8)
    
    fig.canvas.draw_idle()


def run_simulation(enable_plot=True, live_plot=False, save_plot_path="simulation_plot.png"):
    config = load_config()

    mode_mgr = ModeManager(config)
    pitch_ctrl = PitchController(config)
    throttle_ctrl = ThrottleController(config)
    yaw_ctrl = YawController(config)

    nav = NavigationSystem(corrupt_for_test=False)

    # Plotting buffers
    if enable_plot:
        global matplotlib, plt, np
        import matplotlib
        try:
            import tkinter
            matplotlib.use("TkAgg")
        except Exception:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        global hist_time, hist_drone_pos, hist_target_pos_raw, hist_target_pos_ekf
        global hist_drone_speed, hist_target_speed, hist_dist_3d, hist_yaw_error
        global hist_pitch_cmd, hist_yaw_cmd, hist_throttle_cmd, hist_states

        hist_time = []
        hist_drone_pos = []
        hist_target_pos_raw = []
        hist_target_pos_ekf = []
        hist_drone_speed = []
        hist_target_speed = []
        hist_dist_3d = []
        hist_yaw_error = []
        hist_pitch_cmd = []
        hist_yaw_cmd = []
        hist_throttle_cmd = []
        hist_states = []

        start_time = time.time()
        last_plot_time = 0.0
        plot_interval = 0.2  # 5 Hz update rate

        if live_plot:
            plt.ion()
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            fig.suptitle("AVCI DRONE TAKİP SİMÜLASYONU GÖRSELLEŞTİRME (CANLI)", fontsize=14, color='white', weight='bold')
            fig.patch.set_facecolor('#121212')
            
            for ax in axes.flat:
                ax.set_facecolor('#1e1e1e')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                ax.grid(True, color='#333333', linestyle='--')
                for spine in ax.spines.values():
                    spine.set_color('#444444')

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
            else:
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

            # Record plotting data
            if enable_plot:
                t_rel = now - start_time
                hist_time.append(t_rel)
                hist_drone_pos.append(drone_pos)
                hist_target_pos_raw.append(raw_target_pos)
                hist_target_pos_ekf.append(nav_out.get("target_ekf_pos", (0.0, 0.0, 0.0)))
                hist_drone_speed.append(drone_speed)
                hist_target_speed.append(target_speed)
                hist_dist_3d.append(dist_3d)
                hist_yaw_error.append(yaw_error)
                hist_pitch_cmd.append(pitch_cmd)
                hist_yaw_cmd.append(yaw_cmd)
                hist_throttle_cmd.append(target_throttle)
                hist_states.append(state)

                if live_plot and (now - last_plot_time) >= plot_interval and len(hist_time) > 2:
                    last_plot_time = now
                    update_plots(fig, axes, hist_time, hist_drone_pos, hist_target_pos_raw, hist_target_pos_ekf,
                                 hist_drone_speed, hist_target_speed, hist_dist_3d, hist_yaw_error,
                                 hist_pitch_cmd, hist_yaw_cmd, hist_throttle_cmd)
                    plt.pause(0.001)

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
        import traceback
        traceback.print_exc()

    finally:
        drone.set_throttle(0.0)
        drone.set_arm(False)
        drone.disconnect()

        # Save and show final plots
        if enable_plot and len(hist_time) > 2:
            plt.ioff()
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            fig.suptitle("AVCI DRONE TAKİP SİMÜLASYONU GÖRSELLEŞTİRME (ÖZET)", fontsize=14, color="white", weight="bold")
            fig.patch.set_facecolor('#121212')
            
            for ax in axes.flat:
                ax.set_facecolor('#1e1e1e')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                ax.grid(True, color='#333333', linestyle='--')
                for spine in ax.spines.values():
                    spine.set_color('#444444')
                    
            update_plots(fig, axes, hist_time, hist_drone_pos, hist_target_pos_raw, hist_target_pos_ekf,
                         hist_drone_speed, hist_target_speed, hist_dist_3d, hist_yaw_error,
                         hist_pitch_cmd, hist_yaw_cmd, hist_throttle_cmd, final=True)
                         
            plt.tight_layout()
            
            if save_plot_path:
                save_dir = os.path.dirname(save_plot_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                plt.savefig(save_plot_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
                print(f"[!] Grafik başarıyla kaydedildi: {save_plot_path}")
            
            if matplotlib.get_backend() != 'Agg':
                print("[!] Matplotlib penceresi açılıyor. Kapatmak için pencereyi kapatın.")
                plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Avcı Drone Simülasyon Köprüsü")
    parser.add_argument("--no-plot", action="store_true", help="Matplotlib görselleştirmesini kapatır.")
    parser.add_argument("--live", action="store_true", help="Gerçek zamanlı canlı çizim penceresini açar.")
    parser.add_argument("--save-plot", type=str, default="simulation_plot.png", help="Simülasyon bitimindeki grafiğin kaydedileceği yol.")
    args = parser.parse_args()
    
    run_simulation(enable_plot=not args.no_plot, live_plot=args.live, save_plot_path=args.save_plot)