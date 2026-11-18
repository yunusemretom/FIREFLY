"""
Dron sistem tanima: basamak (step) komutlari gonderir, tepkiyi kaydeder.

Isaret kurallarini (pitch+ ileri mi geri mi, roll+ saga mi) ve dinamikleri
(egim acisi / komut, hiz limiti, yaw hizi, dikey hiz) olcmek icindir.

    python analysis/sysid.py --out analysis/data/sysid.csv
"""
import argparse
import csv
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import drone_sdk as drone  # noqa: E402

# (sure_s, throttle, pitch, roll, yaw, etiket)
DEFAULT_SEQ = [
    (3.0, 0.0, 0.0, 0.0, 0.0, "hover"),
    (4.0, 0.0, 0.3, 0.0, 0.0, "pitch+0.3"),
    (4.0, 0.0, 0.0, 0.0, 0.0, "hover"),
    (4.0, 0.0, -0.3, 0.0, 0.0, "pitch-0.3"),
    (4.0, 0.0, 0.0, 0.0, 0.0, "hover"),
    (4.0, 0.0, 0.0, 0.3, 0.0, "roll+0.3"),
    (4.0, 0.0, 0.0, 0.0, 0.0, "hover"),
    (2.0, 0.0, 0.0, 0.0, 0.5, "yaw+0.5"),
    (2.0, 0.0, 0.0, 0.0, 0.0, "hover"),
    (2.0, 0.0, 0.0, 0.0, 1.0, "yaw+1.0"),
    (2.0, 0.0, 0.0, 0.0, 0.0, "hover"),
    (2.0, 0.5, 0.0, 0.0, 0.0, "thr+0.5"),
    (2.0, 0.0, 0.0, 0.0, 0.0, "hover"),
    (2.0, -0.5, 0.0, 0.0, 0.0, "thr-0.5"),
    (3.0, 0.0, 0.0, 0.0, 0.0, "hover"),
]

MAXSPEED_SEQ = [
    (2.0, 0.0, 0.0, 0.0, 0.0, "hover"),
    (8.0, 0.0, 1.0, 0.0, 0.0, "pitch1"),
    (6.0, 0.0, 0.0, 0.0, 0.0, "hover"),
    (8.0, 0.0, 0.6, 0.0, 0.0, "pitch0.6"),
    (6.0, 0.0, 0.0, 0.0, 0.0, "hover"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis", "data", "sysid.csv"))
    ap.add_argument("--maxspeed", action="store_true")
    args = ap.parse_args()
    seq = MAXSPEED_SEQ if args.maxspeed else DEFAULT_SEQ
    if not drone.connect():
        print("baglanti yok")
        return 1
    time.sleep(0.5)
    rows = []
    t0 = time.monotonic()
    try:
        for dur, thr, pit, rol, yaw, lab in seq:
            ts = time.monotonic()
            while time.monotonic() - ts < dur:
                drone.set_control_surfaces(thr, pit, rol, yaw, True)
                tel = drone.get_telemetry()
                d = tel["drone"]
                rows.append([time.monotonic() - t0, lab, thr, pit, rol, yaw,
                             *d["position"], *d["rotation"], *d["velocity"], d["speed"]])
                time.sleep(0.02)
    finally:
        drone.set_control_surfaces(0.0, 0.0, 0.0, 0.0, True)
        time.sleep(0.2)
        drone.disconnect()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "lab", "thr", "pit", "rol", "yaw", "x", "y", "z",
                    "roll", "pitch", "yawang", "vx", "vy", "vz", "speed"])
        w.writerows(rows)
    print(len(rows), "satir ->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
