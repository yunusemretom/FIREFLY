"""
CANLI PURSUIT TESTI v2 — Puruzsuz Takip (drone_sdk)
====================================================
"pursuit_guidance (1).py" canli modunun duzeltilmis surumu. Degisenler:

  1. Zamanlayici tabanli tracker.update KALDIRILDI. Hedef GPS'i artik
     MeasurementGate (change-detection) ile islenir: telemetri alani
     degismedikce filtreye olcum BESLENMEZ. (Eski kod dropout/faz kaymasi
     durumunda bayat olcumu tekrar besliyordu -> hiz/heading salinimi ->
     dalgali yol. Canli modda dalgalanmanin kok nedeni buydu.)
  2. Lead noktasi LeadPointSmoother ile yumusatilir (slew limit + LPF).
  3. Yaw: agir EMA yerine P + gyro-sonumleme (YawController).
  4. Adaptif lead_time: hedef donuste lead otomatik kisalir.

Calistirma:
    python tests/python/test_pursuit_live.py           # grafiksiz (en dusuk jitter)
    python tests/python/test_pursuit_live.py --plot    # canli grafik ile
"""

import argparse
import math
import sys
import os
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.python.navigation.fixed_wing_tracker import FixedWingTracker
from src.python.guidance.pursuit_guidance import (
    SmoothPursuitGuidance, YawController,
)


class LivePursuitRunner:
    # Kontrolcu parametreleri (drone_gui.py ile uyumlu taban degerler)
    KP_YAW = 0.07
    KD_YAW = 0.008          # gyro-sonumleme (bkz. YawController dokumani)
    KP_PITCH = 0.035
    BASE_HOVER_THR = 0.58
    MAX_THR = 1.0
    PITCH_SIGN = -1.0
    PHASE2_START = 4500.0
    DIVE_OFFSET = -25.0

    DT_CTRL = 0.1

    def __init__(self, plot=False):
        self.tracker = FixedWingTracker()
        self.guidance = SmoothPursuitGuidance(self.tracker, lead_time=1.0,
                                              adaptive_lead=True)
        self.yaw_ctrl = YawController(kp=self.KP_YAW, kd=self.KD_YAW)
        self.plot = plot
        self._running = True
        self._roll_smooth = 0.0

    def _connect_sdk(self):
        import drone_sdk as drone
        print("  drone_sdk'ya bağlanılıyor (127.0.0.1:12345) ...")
        while self._running:
            if drone.connect():
                print("  ✓ drone_sdk bağlantısı kuruldu!\n")
                return drone
            print("  ✗ Bağlantı yok — 2 saniye sonra tekrar denenecek ...")
            time.sleep(2.0)
        return None

    # ------------------------------------------------------------------
    def _compute_controls(self, drone_state, aim_info):
        """Aim noktasina gore (yaw, pitch, roll, throttle) uretir."""
        dx, dy, dz = drone_state['pos']
        d_rot = drone_state['rot']
        d_speed = drone_state['speed']
        t_now = drone_state['t']

        ax, ay, az = aim_info['aim_point']
        t_speed = aim_info['target_speed']

        ldx, ldy, ldz = ax - dx, ay - dy, az - dz
        dist_2d = math.hypot(ldx, ldy)
        dist_3d = math.hypot(dist_2d, ldz)

        # --- Yaw: P + gyro damping (EMA yok) ---
        target_yaw = math.degrees(math.atan2(ldy, ldx))
        final_yaw = self.yaw_ctrl.step(t_now, target_yaw, d_rot[2])

        # --- Pitch / Throttle (eski mantik korunuyor) ---
        target_pitch_world = math.degrees(
            math.atan2(ldz, dist_2d if dist_2d > 1e-6 else 1e-6))
        lock_pitch = target_pitch_world

        match_target_thr = (self.BASE_HOVER_THR * 0.92) + (t_speed / 3800.0)
        match_target_thr += ldz * 0.0015

        if dist_3d > self.PHASE2_START:
            target_pitch_drone = lock_pitch + self.DIVE_OFFSET
            final_thr = min(match_target_thr + 0.24, self.MAX_THR)
        else:
            local_scale = max(0.0, dist_3d / self.PHASE2_START)
            flare_trigger_scale = min(1.0, dist_3d / 1000.0)
            flare_effect = (1.0 - flare_trigger_scale) * 15.0
            target_pitch_drone = (lock_pitch + self.DIVE_OFFSET * local_scale
                                  + flare_effect)
            final_thr = match_target_thr + 0.22 * local_scale

        speed_gap = max(0.0, t_speed - d_speed)
        final_thr += min(0.42, speed_gap / 3200.0)
        final_thr = min(self.MAX_THR, final_thr)

        pitch_error = target_pitch_drone - d_rot[1]
        final_pitch = self.PITCH_SIGN * pitch_error * self.KP_PITCH

        # --- Roll: donuse yaslanma + duzelme (eski mantik) ---
        yaw_err = ((target_yaw - d_rot[2] + 180.0) % 360.0) - 180.0
        bank = max(-0.3, min(0.3, yaw_err * 0.012))
        level = -d_rot[0] * 0.15
        self._roll_smooth = 0.50 * self._roll_smooth + 0.25 * (bank + level)

        return {
            'yaw': float(np.clip(final_yaw, -1.0, 1.0)),
            'pitch': float(np.clip(final_pitch, -1.0, 1.0)),
            'roll': float(np.clip(self._roll_smooth, -1.0, 1.0)),
            'throttle': float(np.clip(final_thr, -0.8, self.MAX_THR)),
            'dist_3d': dist_3d,
        }

    # ------------------------------------------------------------------
    def run(self):
        drone = self._connect_sdk()
        if drone is None:
            return

        print("  Pürüzsüz canlı takip başladı. CTRL+C ile durdurun.\n")

        fig = None
        if self.plot:
            fig, artists = self._setup_plot()

        logs = {k: [] for k in ('t', 'dx', 'dy', 'ax', 'ay', 'tx', 'ty',
                                'dist', 'tspd', 'dspd', 'new_gps')}
        t0 = time.time()
        step = 0

        try:
            while self._running:
                step_start = time.time()
                t_now = step_start - t0

                td = drone.get_telemetry()
                d_pos = td["drone"]["position"]
                d_rot = td["drone"]["rotation"]
                d_speed = td["drone"]["speed"]
                tar_pos = td["target"]["position"]

                # Olcum sadece DEGISTIYSE filtreye girer (bayat veri asla)
                is_new = self.guidance.ingest(t_now, tuple(tar_pos))

                if not self.tracker._initialized:
                    time.sleep(self.DT_CTRL)
                    continue

                aim_info = self.guidance.compute_aim(t_now)
                ctl = self._compute_controls(
                    {'pos': d_pos, 'rot': d_rot, 'speed': d_speed, 't': t_now},
                    aim_info)

                drone.set_yaw(ctl['yaw'])
                drone.set_pitch(ctl['pitch'])
                drone.set_throttle(ctl['throttle'])
                drone.set_roll(-ctl['roll'])
                drone.set_arm(True)

                logs['t'].append(t_now)
                logs['dx'].append(d_pos[0]); logs['dy'].append(d_pos[1])
                logs['ax'].append(aim_info['aim_point'][0])
                logs['ay'].append(aim_info['aim_point'][1])
                logs['tx'].append(tar_pos[0]); logs['ty'].append(tar_pos[1])
                logs['dist'].append(ctl['dist_3d'])
                logs['tspd'].append(aim_info['target_speed'])
                logs['dspd'].append(d_speed)
                logs['new_gps'].append(is_new)

                step += 1
                if self.plot and step % 10 == 0:   # 1 Hz cizim: kontrol
                    self._update_plot(fig, artists, logs, d_pos, aim_info)

                if step % 50 == 0:
                    n_new = sum(logs['new_gps'][-50:])
                    print(f"  t={t_now:6.1f}s dist={ctl['dist_3d']:8.0f}cm "
                          f"lead={aim_info['lead_time']:.2f}s "
                          f"hedef_hiz={aim_info['target_speed']:6.0f}cm/s "
                          f"gps(5s)={n_new}")

                elapsed = time.time() - step_start
                sleep_t = self.DT_CTRL - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            print("\n  Canlı takip sonlandırıldı.")
        finally:
            drone.disconnect()
            if self.plot:
                import matplotlib.pyplot as plt
                plt.ioff()
                plt.show()

    # ------------------------------------------------------------------
    def _setup_plot(self):
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d27",
            "axes.edgecolor": "#3a3d4d", "axes.labelcolor": "#c8ccd8",
            "xtick.color": "#7a7d8d", "ytick.color": "#7a7d8d",
            "text.color": "#c8ccd8", "grid.color": "#2a2d3d",
            "grid.linestyle": "--", "grid.alpha": 0.5,
        })
        plt.ion()
        fig, (ax_map, ax_dist) = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle("Pursuit v2 — Pürüzsüz Canlı Takip", fontsize=13,
                     fontweight="bold", color="#e8ecf8")
        artists = {
            'gps': ax_map.plot([], [], 'o', color="#ff5252", ms=4, alpha=0.5,
                               label="Hedef GPS")[0],
            'drone': ax_map.plot([], [], color="#00e676", lw=2,
                                 label="Dron yolu")[0],
            'aim': ax_map.plot([], [], color="#ffab40", lw=1.2, alpha=0.9,
                               label="Aim (yumuşatılmış)")[0],
            'pt_drone': ax_map.plot([], [], '^', color="#00e5ff", ms=9)[0],
            'pt_aim': ax_map.plot([], [], '*', color="#ffab40", ms=13)[0],
            'dist': ax_dist.plot([], [], color="#e040fb", lw=1.5)[0],
            'ax_map': ax_map, 'ax_dist': ax_dist,
        }
        ax_map.set_title("Konum ve Aim Noktası"); ax_map.grid(True)
        ax_map.legend(loc="upper left", facecolor="#1a1d27",
                      edgecolor="#3a3d4d")
        ax_dist.set_title("Aim Noktasına 3D Mesafe (cm)"); ax_dist.grid(True)
        return fig, artists

    def _update_plot(self, fig, artists, logs, d_pos, aim_info):
        artists['gps'].set_data(logs['tx'], logs['ty'])
        artists['drone'].set_data(logs['dx'], logs['dy'])
        artists['aim'].set_data(logs['ax'], logs['ay'])
        artists['pt_drone'].set_data([d_pos[0]], [d_pos[1]])
        artists['pt_aim'].set_data([aim_info['aim_point'][0]],
                                   [aim_info['aim_point'][1]])
        artists['ax_map'].relim(); artists['ax_map'].autoscale_view()
        artists['ax_map'].set_aspect('equal', 'datalim')
        artists['dist'].set_data(logs['t'], logs['dist'])
        artists['ax_dist'].relim(); artists['ax_dist'].autoscale_view()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pürüzsüz canlı pursuit testi")
    parser.add_argument("--plot", action="store_true",
                        help="Canlı grafik (kontrol döngüsüne hafif jitter ekler)")
    args = parser.parse_args()
    LivePursuitRunner(plot=args.plot).run()
