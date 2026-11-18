"""Cevrimdisi testi farkli baslangic geometrileriyle kosar (yaklasma guvenligi)."""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tests", "python"))
sys.path.insert(0, ROOT)
import test_follow_offline as T  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "analysis/data/rec1.csv")
D = np.loadtxt(path, delimiter=",")
p0 = D[0, 24:27]
p1 = D[np.searchsorted(D[:, 0], 1.0), 24:27]
u = (p1 - p0) / np.linalg.norm(p1 - p0)
n = np.array([-u[1], u[0], 0.0])
cases = {
    "arkadan_150m": -15000 * u,
    "kafa_kafaya_150m": 15000 * u,
    "kafa_kafaya_ofset5m": 15000 * u + 500 * n,
    "yandan_150m": 15000 * n,
    "onde_yakin_40m": 4000 * u,
    "arkadan_uzak_300m": -30000 * u,
}
worst = 1e9
for k, off in cases.items():
    off = off.copy()
    off[2] -= 3000
    rows, m = T.run(path, 2000.0, start_offset=tuple(off))
    worst = min(worst, m["en_yakin_mesafe_tum"])
    print(f"{k:22s} en_yakin={m['en_yakin_mesafe_tum']/100:5.1f}m medyan={m['mesafe_medyan']/100:5.1f} "
          f"p95={m['mesafe_p95']/100:5.1f} iz_p95={m['iz_hatasi_p95']/100:4.1f} gecme={m['gecme_sayisi']}")
print(f"EN KOTU en yakin mesafe: {worst/100:.1f} m")
