"""
nisan_canli.py — Drone hedefe DOGRU bakiyor mu, drone_gui.py CALISIRKEN izle.
================================================================================
ONEMLI: Bu script oyuna HIC BAGLANMAZ -> GUI ile cakismaz, link kopmaz, drone
dusmez. drone_gui.py'nin yazdigi /tmp/firefly_ekf.txt dosyasini okur.

KULLANIM:
  1) drone_gui.py calistir, KAMIKAZE MODE ac (GUI ACIK KALSIN).
  2) AYRI terminalde: python3 nisan_canli.py

Gosterir: mesafe | yaw_hata | pitch_hata | hedef kamerada nerede
  yaw_hata ~0  -> drone yatayda hedefe donuk
  pitch_hata ~0 -> drone dikeyde hedefe donuk
  Ikisi de ~0 ise hedef KAMERA MERKEZINDE olmali.
"""
import os, time

DOSYA = "/tmp/firefly_ekf.txt"
print("Nisan canli izleyici (Ctrl+C ile dur). drone_gui.py ACIK olmali.\n")
print(f"{'mesafe':>8} | {'yaw_hata':>8} | {'pitch_hata':>10} | {'hedef nerede':>16} | {'kapanma':>8}")
print("-"*70)

son_ts = None
bekle = 0
try:
    while True:
        try:
            with open(DOSYA) as f:
                p = f.read().strip().split("\t")
            # alanlar: ts,hx,hy,hz,ex,ey,ez,spd,dist,gps,unc,closing,ye,pe,dpitch,dyaw
            if len(p) >= 16:
                ts = float(p[0]); dist = float(p[8]); closing = float(p[11])
                ye = float(p[12]); pe = float(p[13])
                if ts != son_ts:
                    son_ts = ts; bekle = 0
                    yatay = "SAG" if ye > 10 else ("SOL" if ye < -10 else "MERKEZ")
                    dikey = "YUKARI" if pe > 10 else ("ASAGI" if pe < -10 else "ORTA")
                    nerede = f"{yatay} {dikey}"
                    # kapanma: yaklasiyor mu (+) uzaklasiyor mu (-)
                    kap = "yaklas" if closing > 0 else "UZAKLAS"
                    print(f"{dist:7.0f}cm | {ye:+7.1f} | {pe:+9.1f} | {nerede:>16} | {kap:>8}")
                else:
                    bekle += 1
                    if bekle == 25:
                        print("  (veri akmiyor — GUI'de KAMIKAZE MODE acik mi?)")
                        bekle = 0
        except FileNotFoundError:
            print("  (veri yok — drone_gui.py baslat, KAMIKAZE MODE ac)")
            time.sleep(1.0)
        time.sleep(0.2)
except KeyboardInterrupt:
    print("\nDurduruldu.")