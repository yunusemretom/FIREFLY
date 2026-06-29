
import os, time

DOSYA = "/tmp/firefly_ekf.txt"

print("EKF canli izleyici (Ctrl+C ile dur)")
print("drone_gui.py calismiyorsa veya KAMIKAZE MODE kapaliysa veri akmaz.\n")
print(f"{'ham hedef (x,y)':>24} | {'EKF hedef (x,y)':>22} | {'EKF hiz':>11} | "
      f"{'mesafe':>9} | {'yaklasma':>9} | GPS")
print("-" * 95)

son_ts = None
bekleme = 0
try:
    while True:
        try:
            with open(DOSYA) as f:
                parts = f.read().strip().split("\t")
            if len(parts) >= 12:
                (ts, hx, hy, hz, ex, ey, ez, spd, dist, gps, unc, clo) = parts[:12]
                ts = float(ts)
                if ts != son_ts:
                    son_ts = ts
                    bekleme = 0
                    gps_str = "OK  " if gps == "1" else "LOST"
                    print(f"{float(hx):11.0f},{float(hy):10.0f} | "
                          f"{float(ex):11.0f},{float(ey):9.0f} | "
                          f"{float(spd):8.0f}cm/s | "
                          f"{float(dist):7.0f}cm | "
                          f"{float(clo):+7.0f}cm/s | {gps_str}")
                else:
                    bekleme += 1
                    if bekleme == 25:   
                        print("  (veri akmiyor — GUI'de KAMIKAZE MODE acik mi?)")
                        bekleme = 0
        except FileNotFoundError:
            print("  (henuz veri yok — drone_gui.py'i baslatip KAMIKAZE MODE'u ac)")
            time.sleep(1.0)
        time.sleep(0.2)
except KeyboardInterrupt:
    print("\nDurduruldu.")