"""Kayitli telemetri uzerinde TalonEstimator'i oynatir ve gercekle karsilastirir.

    python analysis/replay_estimator.py analysis/data/rec1.csv
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.python.navigation.talon_estimator import TalonEstimator  # noqa: E402


def main(path):
    D = np.loadtxt(path, delimiter=",")
    t = D[:, 0]
    v = D[:, 1:]
    est = TalonEstimator()
    errs, raw_errs, delays, lead_errs = [], [], [], []
    truth = v[:, 23:26]
    cpu = 0.0
    for i in range(len(t)):
        c0 = time.perf_counter()
        est.add_attitude(t[i], *v[i, 14:17])
        est.add_position(t[i], v[i, 11:14])
        cpu = max(cpu, time.perf_counter() - c0)
        if est.ready and t[i] > 10:
            p = est.position()
            errs.append(np.linalg.norm(p - truth[i]))
            raw_errs.append(np.linalg.norm(v[i, 11:14] - truth[i]))
            delays.append(est.delay)
            # 1 s ileri tahmin hatasi
            j = np.searchsorted(t, t[i] + 1.0)
            if j < len(t):
                lead_errs.append(np.linalg.norm(est.predict(t[j] - t[i]) - truth[j]))
    e, r, le = map(np.array, (errs, raw_errs, lead_errs))
    print(f"ham GPS hatasi      : median {np.median(r):6.0f} cm  p95 {np.percentile(r,95):6.0f}  max {r.max():6.0f}")
    print(f"kestirici (simdi)   : median {np.median(e):6.0f} cm  p95 {np.percentile(e,95):6.0f}  max {e.max():6.0f}")
    print(f"kestirici (+1s lead): median {np.median(le):6.0f} cm  p95 {np.percentile(le,95):6.0f}  max {le.max():6.0f}")
    print(f"kestirilen gecikme  : {np.median(delays):.3f} s (min {min(delays):.2f}, max {max(delays):.2f})")
    print(f"en uzun tick CPU    : {cpu*1000:.1f} ms")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "analysis/data/rec1.csv"))
