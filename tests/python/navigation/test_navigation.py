"""
test_navigation.py  —  Navigation modülü doğrulama testi

"""

import os
import sys
import math
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_NAV_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src", "python", "navigation"))
sys.path.insert(0, _NAV_DIR)

from ekf import EKF
from gnss_filter import GNSSCorruptor, GNSSPrefilter
from waypoint_controller import WaypointController


def fake_telemetry(step):
    t = step * 0.05
    drone_pos = [0.0, 0.0, 1000.0]
    target_pos = [1000.0 + 1500.0 * t, 200.0 * math.sin(t), 1500.0]
    return {"drone": {"position": drone_pos}, "target": {"position": target_pos}}


def main():
    drone_ekf = EKF()
    target_ekf = EKF()
    prefilter = GNSSPrefilter()
    corruptor = GNSSCorruptor(enabled=True, seed=42,
                              noise_std=20.0, jump_prob=0.03,
                              jump_magnitude=3000.0, dropout_prob=0.08)
    wp = WaypointController()

    rejected = 0
    dropouts = 0
    total = 200
    errors = []

    for step in range(total):
        tel = fake_telemetry(step)
        true_target = tuple(tel["target"]["position"])

        drone_ekf.step(tuple(tel["drone"]["position"]))

        corrupted = corruptor.corrupt(true_target)
        if corrupted is None:
            dropouts += 1
        meas = prefilter.prefilter(corrupted)
        _, accepted = target_ekf.step(meas)
        if meas is not None and not accepted:
            rejected += 1

        est = target_ekf.get_position()
        err = math.sqrt(sum((a - b) ** 2 for a, b in zip(est, true_target)))
        errors.append(err)

    out = wp.compute(drone_ekf, target_ekf)

    print("=" * 55)
    print("NAVIGATION MODÜLÜ TEST SONUÇLARI")
    print("=" * 55)
    print(f"Toplam adım            : {total}")
    print(f"Dropout (veri kaybı)   : {dropouts}  -> predict-only çalıştı")
    print(f"Reddedilen sıçrama     : {rejected}  (chi-squared gating)")
    print(f"Ort. konum hatası      : {np.mean(errors[20:]):.1f} cm (ilk 20 ısınma hariç)")
    print(f"Medyan konum hatası    : {np.median(errors[20:]):.1f} cm")
    print(f"Maks konum hatası      : {np.max(errors[20:]):.1f} cm")
    print("-" * 55)
    print("WAYPOINT CONTROLLER ÇIKTISI (Meryem'in PID'ine gidecekl taraf):")
    print(f"  Δkonum (dx,dy,dz)    : {out['dx']:.0f}, {out['dy']:.0f}, {out['dz']:.0f} cm")
    print(f"  3D mesafe            : {out['dist_3d']:.0f} cm")
    print(f"  Hedef yaw açısı      : {out['target_yaw']:.1f}°")
    print(f"  Δirtifa açısı (pitch): {out['target_pitch_world']:.1f}°")
    print(f"  Tahmini hedef hızı   : {out['target_speed']:.0f} cm/s")
    print(f"  Filtrelenmiş waypoint: {tuple(round(v) for v in out['waypoint'])}")
    print("=" * 55)

    assert np.median(errors[20:]) < 800, "Medyan hata çok yüksek - EKF tune gerekli"
    assert 1300 < out['target_speed'] < 1700, "Hız tahmini sapmış"
    print("TÜM DOĞRULAMALAR GEÇTİ")


if __name__ == "__main__":
    main()