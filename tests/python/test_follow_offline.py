"""
CEVRIMDISI KAPALI DONGU TESTI — kayitli hedef + olculen dron modeli
====================================================================
Simulatorden kaydedilen hedef telemetrisini (bozuk GPS + yonelim + debug
truth, analysis/record_telemetry.py) oynatir; dronu, analysis/sysid.py ile
olculen basit bir dinamik modelle simule eder; TalonEstimator +
TrailFollowController'i kapali dongude calistirir.

    python tests/python/test_follow_offline.py [kayit.csv] [--follow 2000] [--plot]

Simulator acmadan parametre denemek ve regresyon kontrolu icindir. Canli
sonuclarin yerini TUTMAZ (model basit), ama kaba hatalari yakalar.
"""
import argparse
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.python.navigation.talon_estimator import TalonEstimator  # noqa: E402
from src.python.navigation.gps_target_estimator import GpsTargetEstimator  # noqa: E402
from src.python.navigation.hunter_state import HunterStateEstimator  # noqa: E402
from src.python.guidance.follow_controller import TrailFollowController, DroneModel  # noqa: E402
from src.python.simulation.follow_runner import metrics  # noqa: E402


def vz_from_thr(thr):
    """Olculen kararlı durum throttle -> vz tablosu (interpolasyon)."""
    tab_t = [-1.0, -0.8, -0.5, -0.3, -0.1, -0.05, 0.0, 0.1, 0.2, 1.0]
    tab_v = [-700, -360, 165, 510, 860, 950, 95, 410, 720, 3340]
    if -0.05 < thr < 0.0:     # sureksiz bolge: kaba
        return 950.0
    return float(np.interp(thr, tab_t, tab_v))


class SimDrone:
    def __init__(self, pos, yaw_deg=0.0):
        self.p = np.array(pos, dtype=float)
        self.v = np.zeros(3)
        self.yaw = yaw_deg
        self.tilt = np.zeros(2)     # dunya cercevesinde egim vektoru (deg)

    def step(self, dt, thr, pit, rol, yaw):
        psi = math.radians(self.yaw)
        f = np.array([math.cos(psi), math.sin(psi)])
        r = np.array([math.sin(psi), -math.cos(psi)])
        tilt_cmd = 60.0 * (pit * f + rol * r)
        self.tilt += (1 - math.exp(-dt / 0.15)) * (tilt_cmd - self.tilt)
        v_ss = DroneModel.SPEED_PER_DEG * self.tilt
        n = np.linalg.norm(v_ss)
        if n > 3320:
            v_ss *= 3320 / n
        self.v[:2] += (1 - math.exp(-dt / DroneModel.TAU_V)) * (v_ss - self.v[:2])
        self.v[2] += (1 - math.exp(-dt / 0.6)) * (vz_from_thr(thr) - self.v[2])
        self.yaw += float(np.clip(yaw * 380.0, -256, 256)) * dt
        self.yaw = (self.yaw + 180) % 360 - 180
        self.p += self.v * dt


def run(path, follow=2000.0, start_offset=(-12000.0, 6000.0, -4000.0), dt=0.02,
        competition=True, drone_hz=50.0, true_hunter=False):
    """competition=True: yarisma telemetrisi — hedeften SADECE konum (GPS-only
    kestirici), avcidan GPS + yer hizi (drone_hz ile seyrek) -> HunterStateEstimator.
    competition=False: eski yol (hedef rotation + dron gercek hiz vektoru)."""
    D = np.loadtxt(path, delimiter=",")
    t_rec = D[:, 0]
    v = D[:, 1:]
    est = GpsTargetEstimator() if competition else TalonEstimator()
    ctl = TrailFollowController(follow_distance=follow)
    drone = SimDrone(v[0, 23:26] + np.array(start_offset))
    hunter = HunterStateEstimator()
    next_drone_pkt = 0.0
    cmd = (0.0, 0.0, 0.0, 0.0)
    rows = []
    t = 0.0
    i = 0
    while t < t_rec[-1]:
        while i + 1 < len(t_rec) and t_rec[i + 1] <= t:
            i += 1
        if competition:
            est.tick(t)
            est.add_position(t_rec[i], v[i, 11:14])
            # Avci telemetrisi drone_hz ile; arada komut modeliyle tahmin
            if t >= next_drone_pkt:
                hunter.update_gps(t, drone.p)
                hunter.update_ground_speed(t, float(np.linalg.norm(drone.v[:2])))
                next_drone_pkt = t + 1.0 / drone_hz
            hunter.predict(t)
            p_d, v_d = hunter.position(), hunter.velocity()
            if true_hunter:     # teshis: avci durum kestirimini devre disi birak
                p_d, v_d = drone.p, drone.v
        else:
            est.add_attitude(t_rec[i], *v[i, 14:17])
            est.add_position(t_rec[i], v[i, 11:14])
            p_d, v_d = drone.p, drone.v
        if est.ready and hunter.ready or (est.ready and not competition):
            cmd = ctl.step(est, p_d, v_d, (0.0, 0.0, drone.yaw))
        else:
            cmd = (0.0, 0.0, 0.0, 0.0)
        hunter.set_command(cmd[1], cmd[2], drone.yaw)
        drone.step(dt, *cmd)
        tp = v[i, 23:26]
        row = {"t": t, "dx": drone.p[0], "dy": drone.p[1], "dz": drone.p[2],
               "tx": tp[0], "ty": tp[1], "tz": tp[2], "gx": v[i, 11], "gy": v[i, 12], "gz": v[i, 13]}
        if est.ready:
            row["yaw_err"] = ctl.last["yaw_err"]
            row["mode"] = ctl.last["mode"]
            row["rx"], row["ry"], row["rz"] = ctl.last["p_ref"]
            row["vrx"], row["vry"] = ctl.last["v_ref"][:2]
        rows.append(row)
        t += dt
    return rows, metrics(rows, follow)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=os.path.join(ROOT, "analysis/data/rec1.csv"))
    ap.add_argument("--follow", type=float, default=2000.0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--legacy", action="store_true", help="eski yol: hedef rotation + gercek hiz vektoru")
    ap.add_argument("--drone-hz", type=float, default=50.0)
    a = ap.parse_args()
    rows, m = run(a.path, a.follow, competition=not a.legacy, drone_hz=a.drone_hz)
    for k, val in m.items():
        print(f"  {k:22s}: {val:.2f}" if isinstance(val, float) else f"  {k:22s}: {val}")
    if a.plot:
        from src.python.simulation.follow_runner import plot
        out = os.path.join(ROOT, "analysis/data/offline_follow.png")
        plot(rows, out, a.follow, m)
        print("grafik:", out)
