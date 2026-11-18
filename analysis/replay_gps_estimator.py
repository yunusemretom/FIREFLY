"""Kayitli telemetride GpsTargetEstimator'i (sadece hedef konum) oynatir, gercekle karsilastirir.

    python analysis/replay_gps_estimator.py [kayit.csv] [--window 4] [--delay 1.0]
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.python.navigation.gps_target_estimator import GpsTargetEstimator  # noqa: E402


def load(path):
    """-> (t, hedef_gps(N,3), gercek(N,3)). rec*.csv (ham) veya follow*.csv (baslikli)."""
    with open(path) as f:
        head = f.readline()
    if head[0].isalpha():
        import csv
        R = list(csv.DictReader(open(path)))
        R = [r for r in R if r["tx"] not in ("", "nan")]
        g = lambda k: np.array([float(r[k]) for r in R])  # noqa: E731
        return g("t"), np.column_stack([g("gx"), g("gy"), g("gz")]), \
            np.column_stack([g("tx"), g("ty"), g("tz")])
    D = np.loadtxt(path, delimiter=",")
    return D[:, 0], D[:, 12:15], D[:, 24:27]


def cut_frozen(t, gps, truth, max_still_s=1.0):
    """Sim cokunce gercek konum donar; donmanin basladigi yerden sonrasini at."""
    moving = np.r_[True, np.any(np.abs(np.diff(truth, axis=0)) > 1e-3, axis=1)]
    last_move = t[0]
    for i in range(len(t)):
        if moving[i]:
            last_move = t[i]
        elif t[i] - last_move > max_still_s:
            j = np.searchsorted(t, last_move)
            return t[:j], gps[:j], truth[:j]
    return t, gps, truth


def evaluate(path, follow_lag=1.1, **kw):
    t, gps, truth = cut_frozen(*load(path))
    est = GpsTargetEstimator(**kw)
    e_now, e_lead, e_trail, e_raw = [], [], [], []
    cpu = 0.0
    for i in range(len(t)):
        c0 = time.perf_counter()
        est.tick(t[i])
        est.add_position(t[i], gps[i])
        cpu = max(cpu, time.perf_counter() - c0)
        if est.ready and t[i] > 10:
            e_now.append(np.linalg.norm(est.position() - truth[i]))
            e_raw.append(np.linalg.norm(gps[i] - truth[i]))
            j = np.searchsorted(t, t[i] + 1.0)
            if j < len(t):
                e_lead.append(np.linalg.norm(est.predict(t[j] - t[i]) - truth[j]))
            k = np.searchsorted(t, t[i] - follow_lag)
            e_trail.append(np.linalg.norm(est.trail(t[i] - t[k])[0] - truth[k]))
    return {k: np.array(x) for k, x in
            dict(raw=e_raw, now=e_now, lead1s=e_lead, trail=e_trail).items()}, cpu


def fmt(name, e):
    return f"{name:14s}: median {np.median(e):6.0f} cm  p95 {np.percentile(e, 95):6.0f}  max {e.max():6.0f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=os.path.join(ROOT, "analysis/data/rec1.csv"))
    ap.add_argument("--window", type=float, default=4.0)
    ap.add_argument("--delay", type=float, default=1.0)
    a = ap.parse_args()
    E, cpu = evaluate(a.path, window_s=a.window, delay_s=a.delay)
    for k in ("raw", "now", "lead1s", "trail"):
        print(fmt(k, E[k]))
    print(f"en uzun tick CPU: {cpu * 1000:.1f} ms")
