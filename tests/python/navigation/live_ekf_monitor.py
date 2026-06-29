"""
Live EKF Monitor — Target Tracking Test
========================================
Connects to the simulation server via drone_sdk, feeds target GPS measurements
into TargetEKF, and displays a real-time comparison of:
  - Raw (noisy) GPS measurement
  - EKF filtered estimate
  - Ground truth (only if the server is running in debug mode)

Usage
-----
  python tests/python/navigation/live_ekf_monitor.py
  python tests/python/navigation/live_ekf_monitor.py --host 127.0.0.1 --port 12345
  python tests/python/navigation/live_ekf_monitor.py --no-plot   # terminal only

Controls
--------
  Ctrl+C  — stop and show final statistics
"""

import argparse
import math
import os
import sys
import time
from collections import deque

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import drone_sdk as drone
from src.python.navigation.ekf import TargetEKF

# --- Configuration -----------------------------------------------------------

POLL_HZ = 20          # how fast we read telemetry (the SDK caches it internally)
HISTORY = 500         # number of samples to keep in rolling history
PLOT_INTERVAL = 0.1   # seconds between plot refreshes

# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live EKF target tracking monitor")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=12345)
    p.add_argument("--no-plot", action="store_true", help="Disable matplotlib, terminal only")
    return p.parse_args()


def _dist3(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _fmt(pos) -> str:
    return f"({pos[0]:8.0f}, {pos[1]:8.0f}, {pos[2]:6.0f})"


def run_monitor(host: str, port: int, enable_plot: bool) -> None:
    print(f"[EKF Monitor] Connecting to {host}:{port} …")
    if not drone.connect(host, port):
        print("[EKF Monitor] ERROR: could not connect. Is the simulation running?")
        sys.exit(1)
    print("[EKF Monitor] Connected. Waiting for first telemetry …")

    ekf = TargetEKF()

    # Rolling history buffers  {time, raw_pos, ekf_pos, truth_pos, maha, dropout, spike}
    hist_t        = deque(maxlen=HISTORY)
    hist_raw      = deque(maxlen=HISTORY)
    hist_ekf      = deque(maxlen=HISTORY)
    hist_truth    = deque(maxlen=HISTORY)
    hist_maha     = deque(maxlen=HISTORY)
    hist_err_raw  = deque(maxlen=HISTORY)   # |raw - truth|
    hist_err_ekf  = deque(maxlen=HISTORY)   # |ekf - truth|

    # Matplotlib setup
    fig = ax_xy = ax_z = ax_err = ax_maha = None
    if enable_plot:
        try:
            import matplotlib
            matplotlib.use("TkAgg")   # or "Qt5Agg" — pick what is installed
        except ImportError:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle("Live EKF Target Tracker", fontsize=13)
        ax_xy, ax_z, ax_err, ax_maha = axes.flat

        ax_xy.set_title("XY Trajectory (cm)")
        ax_xy.set_xlabel("X"); ax_xy.set_ylabel("Y")
        ax_xy.set_aspect("equal")

        ax_z.set_title("Altitude over time")
        ax_z.set_xlabel("sample"); ax_z.set_ylabel("Z (cm)")

        ax_err.set_title("Position error vs ground truth (cm)")
        ax_err.set_xlabel("sample"); ax_err.set_ylabel("error (cm)")

        ax_maha.set_title("Mahalanobis distance (innovation)")
        ax_maha.set_xlabel("sample"); ax_maha.set_ylabel("σ")
        ax_maha.axhline(ekf._maha_thresh, color="red", linestyle="--", label="reject thresh")
        ax_maha.legend(fontsize=8)

        plt.ion()
        plt.tight_layout()
        plt.show()

    # ---- main poll loop ----
    t_start = time.time()
    last_plot_t = 0.0
    last_raw = None
    stats_spikes = 0
    stats_dropouts = 0
    sample_idx = 0

    print(
        f"\n{'Sample':>7}  {'Raw position':^32}  {'EKF estimate':^32}  "
        f"{'Maha':>6}  {'Status':<18}"
    )
    print("-" * 110)

    try:
        while True:
            now = time.time()
            tel = drone.get_telemetry()
            raw_pos = tel["target"]["position"]

            # Skip until server sends real data
            if raw_pos == (0.0, 0.0, 0.0) and sample_idx == 0:
                time.sleep(1.0 / POLL_HZ)
                continue

            ekf_pos = ekf.update(raw_pos, timestamp=now)

            # Ground truth (available in debug mode only)
            truth_data = drone.get_debug_truth()
            has_truth = truth_data.get("available", False)
            truth_pos = truth_data["target"]["position"] if has_truth else None

            # Record history
            hist_t.append(now - t_start)
            hist_raw.append(raw_pos)
            hist_ekf.append(ekf_pos)
            hist_truth.append(truth_pos)
            hist_maha.append(ekf.last_maha if ekf.is_ready else 0.0)

            if has_truth and truth_pos:
                hist_err_raw.append(_dist3(raw_pos, truth_pos))
                hist_err_ekf.append(_dist3(ekf_pos, truth_pos))
            else:
                hist_err_raw.append(None)
                hist_err_ekf.append(None)

            if ekf.last_spike_rejected:
                stats_spikes += 1
            if ekf.last_dropout:
                stats_dropouts += 1

            # Terminal output (every sample)
            status = ""
            if not ekf.is_ready:
                status = "INIT"
            elif ekf.last_dropout:
                status = "DROPOUT"
            elif ekf.last_spike_rejected:
                status = "SPIKE REJECTED"
            elif ekf.last_maha > ekf._maha_thresh * 0.7:
                status = f"maha={ekf.last_maha:.1f}"

            corruption = drone.get_active_corruption()
            if corruption:
                status += " | " + ", ".join(corruption[:2])

            print(
                f"{sample_idx:>7}  {_fmt(raw_pos)}  {_fmt(ekf_pos)}  "
                f"{ekf.last_maha:>6.2f}  {status:<18}",
                end="\r",
            )
            if sample_idx % 20 == 0:
                print()   # new line every 20 samples for readability

            sample_idx += 1
            last_raw = raw_pos

            # Matplotlib refresh
            if enable_plot and (now - last_plot_t) >= PLOT_INTERVAL and len(hist_raw) > 2:
                import matplotlib.pyplot as plt
                last_plot_t = now
                _update_plots(
                    fig, ax_xy, ax_z, ax_err, ax_maha,
                    hist_raw, hist_ekf, hist_truth,
                    hist_err_raw, hist_err_ekf, hist_maha,
                    ekf._maha_thresh,
                )
                plt.pause(0.001)

            time.sleep(1.0 / POLL_HZ)

    except KeyboardInterrupt:
        pass

    finally:
        drone.disconnect()
        print("\n\n" + "=" * 60)
        print("  EKF Monitor — Final Statistics")
        print("=" * 60)
        print(f"  Samples collected : {sample_idx}")
        print(f"  Spikes rejected   : {stats_spikes}")
        print(f"  Dropouts detected : {stats_dropouts}")
        if hist_maha:
            valid_maha = [m for m in hist_maha if m > 0]
            if valid_maha:
                print(f"  Mahalanobis  mean : {np.mean(valid_maha):.2f} σ")
                print(f"  Mahalanobis  max  : {np.max(valid_maha):.2f} σ")
        valid_err_raw = [e for e in hist_err_raw if e is not None]
        valid_err_ekf = [e for e in hist_err_ekf if e is not None]
        if valid_err_raw and valid_err_ekf:
            rmse_raw = math.sqrt(np.mean(np.square(valid_err_raw)))
            rmse_ekf = math.sqrt(np.mean(np.square(valid_err_ekf)))
            print(f"  RMSE raw GPS      : {rmse_raw:.1f} cm")
            print(f"  RMSE EKF estimate : {rmse_ekf:.1f} cm")
            improvement = (1.0 - rmse_ekf / rmse_raw) * 100.0 if rmse_raw > 0 else 0.0
            print(f"  Improvement       : {improvement:.1f}%")
        else:
            print("  (Ground truth not available — run server in debug mode for RMSE)")
        print("=" * 60)

        if enable_plot:
            try:
                import matplotlib.pyplot as plt
                plt.ioff()
                plt.show()
            except Exception:
                pass


def _update_plots(
    fig, ax_xy, ax_z, ax_err, ax_maha,
    hist_raw, hist_ekf, hist_truth,
    hist_err_raw, hist_err_ekf, hist_maha,
    maha_thresh,
):
    raw_arr   = np.array(hist_raw)
    ekf_arr   = np.array(hist_ekf)
    maha_arr  = np.array(hist_maha)
    n = len(raw_arr)

    # XY trajectory
    ax_xy.cla()
    ax_xy.set_title("XY Trajectory (cm)")
    ax_xy.plot(raw_arr[:, 0], raw_arr[:, 1], ".", color="gray", ms=2, alpha=0.4, label="raw GPS")
    ax_xy.plot(ekf_arr[:, 0], ekf_arr[:, 1], "-", color="cyan", lw=1.2, label="EKF")
    truth_valid = [(t is not None) for t in hist_truth]
    if any(truth_valid):
        tr = np.array([t for t in hist_truth if t is not None])
        ax_xy.plot(tr[:, 0], tr[:, 1], "--", color="lime", lw=1, label="truth")
    ax_xy.legend(fontsize=7)
    ax_xy.set_aspect("equal", "datalim")

    # Z / altitude
    ax_z.cla()
    ax_z.set_title("Altitude (Z)")
    ax_z.plot(raw_arr[:, 2], color="gray", lw=0.8, label="raw")
    ax_z.plot(ekf_arr[:, 2], color="cyan", lw=1.2, label="EKF")
    ax_z.legend(fontsize=7)

    # Error vs truth
    ax_err.cla()
    ax_err.set_title("Error vs ground truth (cm)")
    valid_r = [(i, e) for i, e in enumerate(hist_err_raw) if e is not None]
    valid_e = [(i, e) for i, e in enumerate(hist_err_ekf) if e is not None]
    if valid_r:
        ix, ey = zip(*valid_r); ax_err.plot(ix, ey, color="gray", lw=0.8, label="raw GPS")
    if valid_e:
        ix, ey = zip(*valid_e); ax_err.plot(ix, ey, color="cyan", lw=1.2, label="EKF")
    if valid_r or valid_e:
        ax_err.legend(fontsize=7)
    else:
        ax_err.text(0.5, 0.5, "No ground truth\n(debug mode off)",
                    ha="center", va="center", transform=ax_err.transAxes, color="gray")

    # Mahalanobis
    ax_maha.cla()
    ax_maha.set_title("Mahalanobis distance (σ)")
    ax_maha.plot(maha_arr, color="orange", lw=0.8)
    ax_maha.axhline(maha_thresh, color="red", linestyle="--", lw=1, label=f"thresh {maha_thresh}")
    ax_maha.legend(fontsize=7)

    fig.canvas.draw_idle()


if __name__ == "__main__":
    args = parse_args()
    run_monitor(args.host, args.port, not args.no_plot)
