"""
Ham telemetri kaydedici (analiz icin).

SDK'nin her parse ettigi satiri, alinis zamaniyla (monotonic) birlikte CSV'ye
yazar. Dron kontrol edilmez (arm edilmez); sadece dinlenir.

    python analysis/record_telemetry.py --seconds 150 --out analysis/data/rec.csv
"""
import argparse
import csv
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import drone_sdk as drone  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "analysis", "data", "rec.csv"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    orig = drone._drone._parse_telemetry

    def hooked(line):
        rows.append((time.monotonic(), line))
        orig(line)

    drone._drone._parse_telemetry = hooked
    if not drone.connect():
        print("baglanti yok")
        return 1
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.seconds:
        time.sleep(0.5)
    drone.disconnect()

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        for t, line in rows:
            w.writerow([f"{t - t0:.4f}"] + line.split(","))
    dur = rows[-1][0] - rows[0][0] if rows else 0
    print(f"{len(rows)} satir, {len(rows) / max(dur, 1e-6):.1f} Hz -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
