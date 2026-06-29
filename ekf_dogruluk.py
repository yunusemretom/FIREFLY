
import time, math, statistics

DOSYA = "/tmp/firefly_ekf.txt"
print("EKF dogruluk izleyici (GUI ACIK olmali, Ctrl+C ile dur)\n")
print(f"{'ham_hata':>9} | {'EKF_hata':>9} | {'iyilesme':>8} | aktif bozma")
print("-"*75)
son=None; ham_l=[]; ekf_l=[]; truth_var=False
try:
    while True:
        try:
            with open(DOSYA) as f:
                p = f.read().strip().split("\t")
            # ...,ye(12),pe(13),dpitch(14),dyaw(15),ham_err(16),ekf_err(17),corr(18+)
            if len(p) >= 18:
                ts=float(p[0])
                if ts!=son:
                    son=ts
                    ham_e=float(p[16]); ekf_e=float(p[17])
                    corr = p[18] if len(p)>18 else ""
                    if ham_e >= 0:   # truth mevcut
                        truth_var=True
                        ham_l.append(ham_e); ekf_l.append(ekf_e)
                        iy = ham_e/ekf_e if ekf_e>1 else 0
                        print(f"{ham_e:8.0f}cm | {ekf_e:8.0f}cm | {iy:7.1f}x | {corr[:50]}")
                    else:
                        print(f"  (debug truth KAPALI — oyunda debug acik degil)")
        except (FileNotFoundError, ValueError, IndexError):
            pass
        time.sleep(0.3)
except KeyboardInterrupt:
    if ham_l:
        hr=math.sqrt(statistics.mean(h*h for h in ham_l))
        er=math.sqrt(statistics.mean(h*h for h in ekf_l))
        print(f"\n=== OZET ({len(ham_l)} olcum) ===")
        print(f"Ham bozuk GPS RMSE: {hr:7.0f} cm")
        print(f"EKF filtreli  RMSE: {er:7.0f} cm")
        print(f"IYILESME: {hr/er:.2f}x")