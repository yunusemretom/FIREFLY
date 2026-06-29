"""
test_ekf.py — Extended Kalman Filter (EKF) Live Dynamic GUI & Unit Tests.

Runs an interactive, live-updating Matplotlib dashboard when executed directly.
Also retains unittest compatibility for automated testing (python test_ekf.py --test).
"""

import os
import sys
import math
import unittest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# Resolve module paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_NAV_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src", "python", "navigation"))
if _NAV_DIR not in sys.path:
    sys.path.insert(0, _NAV_DIR)

try:
    from ekf import EKF
    from gnss_filter import GNSSCorruptor, GNSSPrefilter
except ImportError:
    from src.python.navigation.ekf import EKF
    from src.python.navigation.gnss_filter import GNSSCorruptor, GNSSPrefilter


class TestEKF(unittest.TestCase):
    def test_initialization(self):
        ekf = EKF()
        self.assertIsNotNone(ekf)
        self.assertFalse(ekf.initialized)

    def test_step(self):
        ekf = EKF()
        pos, accepted = ekf.step((100.0, 200.0, 300.0))
        self.assertTrue(ekf.initialized)
        self.assertEqual(pos, (100.0, 200.0, 300.0))


class EKFVisualizerDashboard:
    def __init__(self, max_history=300):
        self.max_history = max_history
        self.reset_simulation()

        # Setup dark modern style
        plt.style.use('dark_background')
        self.fig, self.axs = plt.subplots(2, 2, figsize=(14, 9))
        self.fig.canvas.manager.set_window_title("FIREFLY AVIONICS — EKF REAL-TIME DASHBOARD")
        self.fig.patch.set_facecolor('#0A0A0A')

        for ax_row in self.axs:
            for ax in ax_row:
                ax.set_facecolor('#111111')
                ax.grid(True, linestyle='--', alpha=0.3, color='#444444')
                ax.tick_params(colors='#CCCCCC', labelsize=9)
                for spine in ax.spines.values():
                    spine.set_color('#333333')

        # Layout adjustment for control buttons at bottom
        plt.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.12, wspace=0.25, hspace=0.35)

        self.fig.suptitle(
            "FIREFLY NAVIGATION SYSTEM  |  EXTENDED KALMAN FILTER LIVE DEMO",
            fontsize=14, fontweight='bold', color='#00FF41', y=0.97
        )

        # Setup Plot 1: 2D Trajectory (X vs Y)
        self.ax_traj = self.axs[0, 0]
        self.ax_traj.set_title("Target Trajectory Tracking (X-Y)", color='#00FF41')
        self.ax_traj.set_xlabel("X Position (cm)", color='#AAAAAA')
        self.ax_traj.set_ylabel("Y Position (cm)", color='#AAAAAA')
        self.line_true_2d, = self.ax_traj.plot([], [], 'g--', label="True Path", alpha=0.8, linewidth=1.5)
        self.scat_gnss_2d, = self.ax_traj.plot([], [], 'r.', label="Raw GNSS", alpha=0.4, markersize=4)
        self.line_ekf_2d, = self.ax_traj.plot([], [], 'c-', label="EKF Estimated", linewidth=2.0)
        self.head_true, = self.ax_traj.plot([], [], 'go', markersize=7, label="True Pos")
        self.head_ekf, = self.ax_traj.plot([], [], 'co', markersize=8, markeredgecolor='white', label="EKF Pos")
        self.ax_traj.legend(loc="upper left", facecolor='#1A1A1A', edgecolor='#333')

        # Setup Plot 2: Altitude vs Time (Z vs Step)
        self.ax_alt = self.axs[0, 1]
        self.ax_alt.set_title("Altitude Profile (Z vs Time)", color='#00FF41')
        self.ax_alt.set_xlabel("Time Step", color='#AAAAAA')
        self.ax_alt.set_ylabel("Z Altitude (cm)", color='#AAAAAA')
        self.line_true_z, = self.ax_alt.plot([], [], 'g--', label="True Z", alpha=0.8)
        self.scat_gnss_z, = self.ax_alt.plot([], [], 'r.', label="Raw GNSS Z", alpha=0.4, markersize=4)
        self.line_ekf_z, = self.ax_alt.plot([], [], 'c-', label="EKF Z", linewidth=2.0)
        self.ax_alt.legend(loc="upper left", facecolor='#1A1A1A', edgecolor='#333')

        # Setup Plot 3: Position Error Over Time
        self.ax_err = self.axs[1, 0]
        self.ax_err.set_title("Real-Time Position Error (cm)", color='#00FF41')
        self.ax_err.set_xlabel("Time Step", color='#AAAAAA')
        self.ax_err.set_ylabel("3D Error (cm)", color='#AAAAAA')
        self.line_err, = self.ax_err.plot([], [], '#FF0055', linewidth=1.5)
        self.txt_stats = self.ax_err.text(
            0.03, 0.92, "", transform=self.ax_err.transAxes,
            color='#00FF41', fontfamily='monospace', fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#1A1A1A', alpha=0.85, edgecolor='#00FF41')
        )

        # Setup Plot 4: Speed Estimation
        self.ax_spd = self.axs[1, 1]
        self.ax_spd.set_title("Target Speed Estimation (cm/s)", color='#00FF41')
        self.ax_spd.set_xlabel("Time Step", color='#AAAAAA')
        self.ax_spd.set_ylabel("Speed (cm/s)", color='#AAAAAA')
        self.line_true_spd, = self.ax_spd.plot([], [], 'g--', label="True Speed", alpha=0.8)
        self.line_ekf_spd, = self.ax_spd.plot([], [], '#FFD700', label="EKF Est Speed", linewidth=1.8)
        self.ax_spd.legend(loc="upper left", facecolor='#1A1A1A', edgecolor='#333')

        # Buttons layout
        ax_btn_pause = plt.axes([0.15, 0.02, 0.15, 0.05])
        ax_btn_reset = plt.axes([0.42, 0.02, 0.15, 0.05])
        ax_btn_noise = plt.axes([0.69, 0.02, 0.18, 0.05])

        self.btn_pause = Button(ax_btn_pause, 'Pause / Play', color='#222222', hovercolor='#333333')
        self.btn_reset = Button(ax_btn_reset, 'Reset Simulation', color='#222222', hovercolor='#333333')
        self.btn_noise = Button(ax_btn_noise, 'Toggle GNSS Noise: ON', color='#1A331A', hovercolor='#2A442A')

        for b in [self.btn_pause, self.btn_reset, self.btn_noise]:
            b.label.set_color('#00FF41')
            b.label.set_fontweight('bold')

        self.btn_pause.on_clicked(self.toggle_pause)
        self.btn_reset.on_clicked(lambda event: self.reset_simulation())
        self.btn_noise.on_clicked(self.toggle_noise)

        self.is_paused = False

    def reset_simulation(self):
        self.ekf = EKF(dt=0.05)
        self.corruptor = GNSSCorruptor(enabled=True, noise_std=20.0, jump_prob=0.03, jump_magnitude=3000.0, dropout_prob=0.08)
        self.prefilter = GNSSPrefilter()

        self.step_idx = 0
        self.history_steps = []
        self.history_true_pos = []
        self.history_gnss_pos = []
        self.history_ekf_pos = []
        self.history_true_spd = []
        self.history_ekf_spd = []
        self.history_errors = []

        self.dropouts = 0
        self.rejected = 0

    def toggle_pause(self, event):
        self.is_paused = not self.is_paused

    def toggle_noise(self, event):
        self.corruptor.enabled = not self.corruptor.enabled
        status = "ON" if self.corruptor.enabled else "OFF"
        self.btn_noise.label.set_text(f"Toggle GNSS Noise: {status}")
        color = '#1A331A' if self.corruptor.enabled else '#331A1A'
        self.btn_noise.ax.set_facecolor(color)
        self.fig.canvas.draw_idle()

    def get_target_telemetry(self, step):
        t = step * 0.05
        # Dynamic spiral trajectory
        x = 1000.0 + 1500.0 * math.sin(0.3 * t) + 200.0 * math.cos(1.2 * t)
        y = 200.0 + 1200.0 * math.cos(0.3 * t) + 150.0 * math.sin(1.2 * t)
        z = 1500.0 + 400.0 * math.sin(0.15 * t)

        # True speed calculation
        vx = 1500.0 * 0.3 * math.cos(0.3 * t) - 200.0 * 1.2 * math.sin(1.2 * t)
        vy = -1200.0 * 0.3 * math.sin(0.3 * t) + 150.0 * 1.2 * math.cos(1.2 * t)
        vz = 400.0 * 0.15 * math.cos(0.15 * t)
        spd = math.sqrt(vx**2 + vy**2 + vz**2)

        return (x, y, z), spd

    def update(self, frame):
        if self.is_paused:
            return

        self.step_idx += 1
        true_pos, true_spd = self.get_target_telemetry(self.step_idx)

        # Corrupt GNSS measurement
        corrupted = self.corruptor.corrupt(true_pos)
        if corrupted is None:
            self.dropouts += 1
        meas = self.prefilter.prefilter(corrupted)

        # EKF Step
        est_pos, accepted = self.ekf.step(meas)
        if meas is not None and not accepted:
            self.rejected += 1

        est_spd = self.ekf.get_speed()
        err = math.sqrt(sum((a - b)**2 for a, b in zip(est_pos, true_pos)))

        # Record history
        self.history_steps.append(self.step_idx)
        self.history_true_pos.append(true_pos)
        self.history_gnss_pos.append(corrupted if corrupted else (np.nan, np.nan, np.nan))
        self.history_ekf_pos.append(est_pos)
        self.history_true_spd.append(true_spd)
        self.history_ekf_spd.append(est_spd)
        self.history_errors.append(err)

        # Limit history for smooth display performance
        if len(self.history_steps) > self.max_history:
            self.history_steps.pop(0)
            self.history_true_pos.pop(0)
            self.history_gnss_pos.pop(0)
            self.history_ekf_pos.pop(0)
            self.history_true_spd.pop(0)
            self.history_ekf_spd.pop(0)
            self.history_errors.pop(0)

        # Convert lists to numpy for plotting
        steps_arr = np.array(self.history_steps)
        true_arr = np.array(self.history_true_pos)
        gnss_arr = np.array(self.history_gnss_pos)
        ekf_arr = np.array(self.history_ekf_pos)

        # Update 2D Trajectory
        self.line_true_2d.set_data(true_arr[:, 0], true_arr[:, 1])
        self.scat_gnss_2d.set_data(gnss_arr[:, 0], gnss_arr[:, 1])
        self.line_ekf_2d.set_data(ekf_arr[:, 0], ekf_arr[:, 1])
        self.head_true.set_data([true_pos[0]], [true_pos[1]])
        self.head_ekf.set_data([est_pos[0]], [est_pos[1]])
        self.ax_traj.relim()
        self.ax_traj.autoscale_view()

        # Update Altitude Profile
        self.line_true_z.set_data(steps_arr, true_arr[:, 2])
        self.scat_gnss_z.set_data(steps_arr, gnss_arr[:, 2])
        self.line_ekf_z.set_data(steps_arr, ekf_arr[:, 2])
        self.ax_alt.relim()
        self.ax_alt.autoscale_view()

        # Update Position Error
        self.line_err.set_data(steps_arr, self.history_errors)
        self.ax_err.relim()
        self.ax_err.autoscale_view()

        # Update Stats Overlay
        recent_errs = self.history_errors[max(0, len(self.history_errors) - 100):]
        mean_e = np.mean(recent_errs) if recent_errs else 0.0
        median_e = np.median(recent_errs) if recent_errs else 0.0
        max_e = np.max(recent_errs) if recent_errs else 0.0

        stats_str = (
            f"Step      : {self.step_idx}\n"
            f"Curr Err  : {err:.1f} cm\n"
            f"Mean Err  : {mean_e:.1f} cm\n"
            f"Median Err: {median_e:.1f} cm\n"
            f"Max Err   : {max_e:.1f} cm\n"
            f"Dropouts  : {self.dropouts}\n"
            f"Rejections: {self.rejected}"
        )
        self.txt_stats.set_text(stats_str)

        # Update Speed Tracking
        self.line_true_spd.set_data(steps_arr, self.history_true_spd)
        self.line_ekf_spd.set_data(steps_arr, self.history_ekf_spd)
        self.ax_spd.relim()
        self.ax_spd.autoscale_view()

    def start(self):
        self.anim = FuncAnimation(self.fig, self.update, interval=30, cache_frame_data=False)
        plt.show()


def run_gui():
    dashboard = EKFVisualizerDashboard()
    dashboard.start()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        unittest.main(argv=[sys.argv[0]])
    else:
        run_gui()
