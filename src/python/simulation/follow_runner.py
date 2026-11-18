"""
CANLI IZ TAKIP KOSUCUSU — yarisma modu (GPS-only) + TrailFollowController
=================================================================
Tek ana dongu, 50 Hz, tek TCP satiri (set_control_surfaces).

    python -m src.python.simulation.follow_runner --seconds 120 --follow 2000

Debug truth aciksa basari metriklerini (gercek mesafe, iz hatasi, hedefi
gecme sayisi) hesaplar ve analysis/data/ altina CSV + PNG yazar. Debug truth
YALNIZCA olcum icindir; kontrol dongusune hicbir sekilde girmez.
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import deque

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

import drone_sdk as drone  # noqa: E402
from src.python.navigation.talon_estimator import TalonEstimator  # noqa: E402
from src.python.navigation.gps_target_estimator import GpsTargetEstimator  # noqa: E402
from src.python.navigation.hunter_state import HunterStateEstimator  # noqa: E402
from src.python.simulation.competition_telemetry import CompetitionTelemetry  # noqa: E402
from src.python.guidance.follow_controller import TrailFollowController  # noqa: E402

DT = 0.02


def start_frame_recorder(out_dir, period=0.25, t0=None):
    """Ayri is parcaciginda oyun karelerini + o anki telemetriyi kaydeder.

    Kalibrasyon / model testi icin. Kontrol dongusunu yavaslatmaz.
    """
    import json
    import threading
    import cv2
    from src.python.vision.window_capture import WindowCapture

    os.makedirs(out_dir, exist_ok=True)
    cap = WindowCapture()
    stop = threading.Event()
    meta = open(os.path.join(out_dir, "meta.jsonl"), "w")

    def loop():
        i = 0
        while not stop.is_set():
            ts = time.monotonic()
            f = cap.grab()
            tel = drone.get_telemetry()
            tr = drone.get_debug_truth()
            name = f"f{i:05d}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
            meta.write(json.dumps({
                "file": name, "t": ts - (t0 or 0.0),
                "drone_pos": tel["drone"]["position"], "drone_rot": tel["drone"]["rotation"],
                "target_rot": tel["target"]["rotation"],
                "truth_target": tr["target"]["position"] if tr["available"] else None,
            }) + "\n")
            meta.flush()
            i += 1
            time.sleep(max(0.0, period - (time.monotonic() - ts)))

    th = threading.Thread(target=loop, daemon=True)
    th.start()

    def stopper():
        stop.set()
        th.join(timeout=1.0)
        meta.close()
    return stopper


def run(seconds, follow, alt_offset, out_prefix, quiet=False, frames_dir=None, min_sep=1200.0,
        frame_period=0.25, legacy=False, drone_hz=50.0):
    """legacy=False (varsayilan): YARISMA MODU — kontrol yolu yalnizca sartnamedeki
    telemetriyi gorur (CompetitionTelemetry): hedeften konum (<=5 Hz), avcidan
    GPS + yer hizi + durus (drone_hz). legacy=True: eski yol (hedef rotation +
    dron hiz vektoru; yarismada YOK)."""
    est = TalonEstimator() if legacy else GpsTargetEstimator()
    comp = CompetitionTelemetry(drone_hz=drone_hz)
    hunter = HunterStateEstimator()
    ctl = TrailFollowController(follow_distance=follow, altitude_offset=alt_offset,
                                min_separation=min_sep)

    if not drone.connect():
        print("baglanti yok")
        return None
    time.sleep(0.3)

    rows = []
    truth_hist = deque(maxlen=5000)
    t0 = time.monotonic()
    next_t = t0
    stop_frames = start_frame_recorder(frames_dir, period=frame_period, t0=t0) if frames_dir else None
    worst_dt = 0.0
    last_print = 0.0
    last_change_t = 0.0
    last_drone_pos = None
    frozen = False
    try:
        while True:
            now = time.monotonic()
            t = now - t0
            if t > seconds:
                break
            tel = drone.get_telemetry()
            d = tel["drone"]
            tg = tel["target"]
            # Guvenlik: dron telemetrisi ~100 Hz gelir; 1 s degismezse
            # simulator donmus/cokmus demektir -> dongu durur.
            if d["position"] != last_drone_pos:
                last_drone_pos = d["position"]
                last_change_t = t
            elif t - last_change_t > 1.0:
                print(f"!! t={t:.1f}: dron telemetrisi 1 s'dir degismiyor (simulator dondu/coktu?) -> durduruluyor")
                frozen = True
                break
            if legacy:
                est.add_attitude(t, *tg["rotation"])
                est.add_position(t, tg["position"])
                p_d, v_d, rot_d = d["position"], d["velocity"], d["rotation"]
                ok = est.ready
            else:
                c = comp.update(t, tel)
                cd, ct = c["drone"], c["target"]
                est.tick(t)
                if ct is not None and ct["fresh"]:
                    est.add_position(t, ct["position"], ct["speed"])
                if cd["fresh"]:
                    hunter.update_gps(t, cd["position"])
                    hunter.update_ground_speed(t, cd["ground_speed"])
                hunter.predict(t)
                rot_d = cd["rotation"]
                ok = est.ready and hunter.ready
                if hunter.ready:
                    p_d, v_d = hunter.position(), hunter.velocity()

            if ok:
                thr, pit, rol, yaw = ctl.step(est, p_d, v_d, rot_d)
            else:
                thr, pit, rol, yaw = 0.0, 0.0, 0.0, 0.0
            if not legacy:
                hunter.set_command(pit, rol, rot_d[2])
            drone.set_control_surfaces(thr, pit, rol, yaw, True)

            tr = drone.get_debug_truth()
            tpos = np.array(tr["target"]["position"]) if tr["available"] else None
            if tpos is not None:
                truth_hist.append((t, tpos))
            row = {
                "t": t, "dx": d["position"][0], "dy": d["position"][1], "dz": d["position"][2],
                "gx": tg["position"][0], "gy": tg["position"][1], "gz": tg["position"][2],
                "groll": tg["rotation"][0], "gpitch": tg["rotation"][1], "gyaw": tg["rotation"][2],
                "dvx": d["velocity"][0], "dvy": d["velocity"][1], "dvz": d["velocity"][2],
                "droll": d["rotation"][0], "dpitch": d["rotation"][1], "dyaw": d["rotation"][2],
                "thr": thr, "pit": pit, "rol": rol, "yaw": yaw,
                "ready": ok, "delay": est.delay if est.delay is not None else float("nan"),
            }
            if not legacy and hunter.ready:
                row.update({"hx": p_d[0], "hy": p_d[1], "hvx": v_d[0], "hvy": v_d[1]})
            if ok:
                pe = est.position()
                row.update({"ex": pe[0], "ey": pe[1], "ez": pe[2],
                            "rx": ctl.last["p_ref"][0], "ry": ctl.last["p_ref"][1],
                            "rz": ctl.last["p_ref"][2], "yaw_err": ctl.last["yaw_err"],
                            "mode": ctl.last["mode"]})
            if tpos is not None:
                row.update({"tx": tpos[0], "ty": tpos[1], "tz": tpos[2]})
            rows.append(row)

            if not quiet and t - last_print > 2.0 and ok:
                last_print = t
                true_d = np.linalg.norm(tpos - np.array(d["position"])) if tpos is not None else float("nan")
                print(f"t={t:6.1f} ref_hata={ctl.last['err_ref']:6.0f} "
                      f"hedef_mesafe(kest)={ctl.last['dist_target']:6.0f} gercek={true_d:6.0f} "
                      f"hiz={d['speed']:5.0f} yaw_err={ctl.last['yaw_err']:5.1f} "
                      f"gecikme={est.delay:.2f} mod={ctl.last['mode']}")

            next_t += DT
            sl = next_t - time.monotonic()
            if sl > 0:
                time.sleep(sl)
            else:
                worst_dt = max(worst_dt, -sl)
                next_t = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        if stop_frames:
            stop_frames()
        drone.set_control_surfaces(0.0, 0.0, 0.0, 0.0, True)
        time.sleep(0.1)
        drone.disconnect()

    if frozen:
        # Donma oncesi son ~1 s'lik veri bayat (truth da donar) -> metriklerden at
        rows = [r for r in rows if r["t"] < last_change_t - 0.1]
    if out_prefix:
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with open(out_prefix + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    m = metrics(rows, follow)
    m["loop_overrun_max_ms"] = worst_dt * 1000
    if out_prefix:
        plot(rows, out_prefix + ".png", follow, m)
    return m


def metrics(rows, follow, settle_s=15.0):
    R = [r for r in rows if "tx" in r and r["t"] > settle_s]
    if not R:
        return {}
    t = np.array([r["t"] for r in R])
    D = np.array([[r["dx"], r["dy"], r["dz"]] for r in R])
    T = np.array([[r["tx"], r["ty"], r["tz"]] for r in R])
    dist = np.linalg.norm(T - D, axis=1)
    # Gercek iz noktasi: gercek hedefin (follow/V) s onceki konumu
    allT = np.array([[r["t"], r["tx"], r["ty"], r["tz"]] for r in rows if "tx" in r])
    vel = np.gradient(allT[:, 1:3], allT[:, 0], axis=0)
    V = np.median(np.linalg.norm(vel, axis=1))
    lag = follow / V
    ideal = np.column_stack([np.interp(t - lag, allT[:, 0], allT[:, i]) for i in (1, 2, 3)])
    trail_err = np.linalg.norm(ideal - D, axis=1)
    # Hedefin onune gecme: (dron - hedef) . hedef_hiz > 0
    vT = np.column_stack([np.interp(t, allT[:, 0], vel[:, i]) for i in (0, 1)])
    ahead = np.sum((D[:, :2] - T[:, :2]) * vT, axis=1) > 0
    n_pass = int(np.sum(np.diff(ahead.astype(int)) == 1))
    # Kamera hizasi
    ye = np.array([abs(r.get("yaw_err", np.nan)) for r in R])
    Aall = [r for r in rows if "tx" in r and r.get("ready", True)]
    dmin = min(math.dist((r["dx"], r["dy"], r["dz"]), (r["tx"], r["ty"], r["tz"])) for r in Aall)
    return {
        "sure_s": float(t[-1] - t[0]),
        "en_yakin_mesafe_tum": float(dmin),
        "mesafe_medyan": float(np.median(dist)), "mesafe_p5": float(np.percentile(dist, 5)),
        "mesafe_p95": float(np.percentile(dist, 95)),
        "iz_hatasi_medyan": float(np.median(trail_err)),
        "iz_hatasi_p95": float(np.percentile(trail_err, 95)),
        "hedefin_onunde_%": float(100 * ahead.mean()), "gecme_sayisi": n_pass,
        "yaw_hata_medyan": float(np.nanmedian(ye)), "yaw_hata_p95": float(np.nanpercentile(ye, 95)),
        "hedef_hizi": float(V),
    }


def plot(rows, path, follow, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = [r for r in rows if "tx" in r]
    if not R:
        return
    t = np.array([r["t"] for r in R])
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].plot([r["tx"] for r in R], [r["ty"] for r in R], "-", color="#d33", lw=1, label="Hedef (gercek)")
    G = [(r["gx"], r["gy"]) for r in R]
    ax[0].plot(*zip(*G), ".", color="#f99", ms=2, label="Hedef GPS (bozuk)")
    ax[0].plot([r["dx"] for r in R], [r["dy"] for r in R], "-", color="#06c", lw=1.2, label="Dron")
    ax[0].set_aspect("equal", "datalim")
    ax[0].legend()
    ax[0].set_title("Yol (ust gorunum)")
    D = np.array([[r["dx"], r["dy"], r["dz"]] for r in R])
    T = np.array([[r["tx"], r["ty"], r["tz"]] for r in R])
    ax[1].plot(t, np.linalg.norm(T - D, axis=1) / 100, label="Gercek mesafe (m)")
    ax[1].plot(t, (T[:, 2] - D[:, 2]) / 100, label="Irtifa farki (m)")
    ax[1].axhline(follow / 100, color="k", ls="--", lw=0.8, label="Hedef takip mesafesi")
    ax[1].set_xlabel("t (s)")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    ax[1].set_title("medyan %.1f m | iz hatasi medyan %.1f m | gecme %d" % (
        m.get("mesafe_medyan", 0) / 100, m.get("iz_hatasi_medyan", 0) / 100, m.get("gecme_sayisi", -1)))
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--follow", type=float, default=2000.0, help="takip mesafesi (cm)")
    ap.add_argument("--alt", type=float, default=0.0, help="hedefe gore irtifa ofseti (cm)")
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis", "data", "follow"))
    ap.add_argument("--frames", default=None, help="kare+telemetri kayit dizini (kalibrasyon)")
    ap.add_argument("--minsep", type=float, default=1200.0, help="carpisma onleme yaricapi (cm)")
    ap.add_argument("--frame-period", type=float, default=0.25)
    ap.add_argument("--legacy", action="store_true",
                    help="eski yol: hedef rotation + dron hiz vektoru (yarismada YOK)")
    ap.add_argument("--drone-hz", type=float, default=50.0,
                    help="avci telemetri hizi (sartname: 3.125-50 Hz)")
    a = ap.parse_args()
    res = run(a.seconds, a.follow, a.alt, a.out, frames_dir=a.frames, min_sep=a.minsep,
              frame_period=a.frame_period, legacy=a.legacy, drone_hz=a.drone_hz)
    if res:
        print("\n=== METRIKLER (t>15 s, debug truth ile) ===")
        for k, v in res.items():
            print(f"  {k:22s}: {v:.2f}" if isinstance(v, float) else f"  {k:22s}: {v}")
