"""
CANLI-KOSUL PURSUIT KARSILASTIRMA HARNESS'I
============================================
Amac: "Sadece canli modda dalgali yol" sorununu masada yeniden uretmek ve
cozum adimlarinin etkisini SAYIYLA gostermek.

Canli ortamin sim'de olmayan ozellikleri burada birebir taklit edilir:
  - Hedef GPS'i kontrol dongusuyle SENKRON DEGIL: 1 Hz + faz kaymasi + jitter
  - Dropout: paket %15 olasilikla hic gelmez (telemetri alani bayat kalir)
  - Telemetri alani her zaman SON GELEN degeri tutar (stale hold)
  - Kontrol dongusu 10 Hz ama zamanlama jitter'li

Karsilastirilan konfigurasyonlar:
  A) baseline : eski kod — zamanlayici tabanli update (bayat olcum beslenir),
                ham lead point, P-only yaw + agir EMA (0.7/0.3)
  B) gate     : SADECE olcum kapisi eklenir (change-detection update)
  C) full     : gate + lead-point yumusatma + PD yaw + adaptif lead_time

Calistirma:
    python tests/python/test_pursuit_smoothing.py            # metrik tablosu
    python tests/python/test_pursuit_smoothing.py --plot     # + yol grafigi
"""

import argparse
import math
import sys

import numpy as np

sys.path.insert(0, '.')
from src.python.navigation.fixed_wing_tracker import FixedWingTracker
from src.python.guidance.pursuit_guidance import (
    SmoothPursuitGuidance, YawController, wrap_deg,
)

# --- Senaryo sabitleri --------------------------------------------------
TARGET_SPEED = 1800.0      # cm/s
TARGET_RADIUS = 25000.0    # cm (dairesel yorunge)
TARGET_Z = 8000.0
GPS_SIGMA = (1500.0, 1500.0, 400.0)
GPS_PERIOD = 1.0
GPS_PHASE = 0.37           # dongusuyle senkron olmayan gelis fazi
GPS_JITTER = 0.05
DROPOUT_PROB = 0.15

DT_CTRL = 0.1
LOOP_JITTER = 0.02
T_END = 60.0

DRONE_START = np.array([0.0, -10000.0, 8000.0])
DRONE_SPEED = 2600.0       # cm/s (sabit; dalgalanma analizi icin yeterli)
MAX_YAW_RATE = 100.0       # deg/s (stick=1.0 karsiligi)
CAPTURE_RADIUS = 3000.0    # cm

KP_YAW_OLD = 0.07          # eski kontrolcu kazanci (drone_gui ile ayni)


def target_true(t):
    w = TARGET_SPEED / TARGET_RADIUS
    return np.array([TARGET_RADIUS * math.cos(w * t),
                     TARGET_RADIUS * math.sin(w * t),
                     TARGET_Z])


def build_gps_stream(rng, t_end):
    """(gelis_zamani, gurultulu_konum) listesi. Ilk paket t=0'da gelir."""
    stream = [(0.0, target_true(0.0) + rng.normal(0, GPS_SIGMA))]
    k = 1
    while True:
        t_arr = GPS_PHASE + k * GPS_PERIOD + rng.uniform(-GPS_JITTER, GPS_JITTER)
        k += 1
        if t_arr > t_end:
            break
        if rng.random() < DROPOUT_PROB:
            continue  # dropout: telemetri alani bayat kalir
        stream.append((t_arr, target_true(t_arr) + rng.normal(0, GPS_SIGMA)))
    return stream


def run_once(seed, mode):
    """Tek kosum; dalgalanma ve yakalama metriklerini dondurur."""
    rng = np.random.default_rng(seed)
    gps_stream = build_gps_stream(rng, T_END)
    gps_idx = 0
    telemetry_field = gps_stream[0][1]  # stale-hold alani

    tracker = FixedWingTracker()
    guidance = SmoothPursuitGuidance(tracker, lead_time=1.0,
                                     adaptive_lead=(mode == 'full'))
    yaw_ctrl = YawController(kp=KP_YAW_OLD)  # kd: modul varsayilani (gyro damping)

    pos = DRONE_START.copy()
    heading = 90.0            # baslangicta +Y yonune bakiyor
    yaw_smooth = 0.0          # eski EMA durumu
    gps_counter = 0.0         # eski zamanlayici durumu

    t = 0.0
    headings, sticks, aim_jumps, dists = [], [], [], []
    prev_aim = None
    capture_t = None
    capture_idx = None
    path = [pos[:2].copy()]

    while t < T_END:
        dt = DT_CTRL + rng.uniform(-LOOP_JITTER, LOOP_JITTER)

        # Telemetri alanini guncelle (gelen paket varsa)
        while gps_idx + 1 < len(gps_stream) and gps_stream[gps_idx + 1][0] <= t:
            gps_idx += 1
            telemetry_field = gps_stream[gps_idx][1]

        # --- Filtre guncelleme stratejisi ---
        if mode == 'baseline':
            gps_counter += dt
            if gps_counter >= GPS_PERIOD - 1e-9 or not tracker._initialized:
                gps_counter = 0.0
                tracker.update(t, *telemetry_field)   # bayat olabilir!
        else:
            guidance.ingest(t, tuple(telemetry_field))

        # --- Nisan noktasi ---
        if not tracker._initialized:
            t += dt
            continue
        aim_info = guidance.compute_aim(t)
        aim = aim_info['aim_point'] if mode == 'full' else aim_info['raw_lead_point']
        aim = np.asarray(aim)

        # --- Yaw kontrolu ---
        bearing = math.degrees(math.atan2(aim[1] - pos[1], aim[0] - pos[0]))
        if mode == 'full':
            stick = yaw_ctrl.step(t, bearing, heading)
        else:
            err = wrap_deg(bearing - heading)
            yaw_smooth = 0.7 * yaw_smooth + 0.3 * (KP_YAW_OLD * err)
            stick = float(np.clip(yaw_smooth, -1.0, 1.0))

        # --- Drone kinematigi ---
        heading = (heading + stick * MAX_YAW_RATE * dt) % 360.0
        vz = float(np.clip((aim[2] - pos[2]) * 0.8, -500.0, 500.0))
        v_xy = math.sqrt(max(DRONE_SPEED ** 2 - vz ** 2, 0.0))
        pos += np.array([v_xy * math.cos(math.radians(heading)),
                         v_xy * math.sin(math.radians(heading)),
                         vz]) * dt

        # --- Kayit ---
        headings.append(heading)
        sticks.append(stick)
        if prev_aim is not None:
            aim_jumps.append(np.linalg.norm(aim - prev_aim) / dt)
        prev_aim = aim
        dist = np.linalg.norm(pos - target_true(t))
        dists.append(dist)
        if capture_t is None and dist < CAPTURE_RADIUS:
            capture_t = t
            capture_idx = len(dists) - 1
        path.append(pos[:2].copy())
        t += dt

    # Dalgalanma metrikleri YAKLASMA fazina aittir (yakalamadan oncesi);
    # yakalama sonrasi yakin-mesafe kovalamacasi ayri bir problem oldugu
    # icin (terminal guidance) metrikleri kirletmemeli.
    end = capture_idx if capture_idx is not None else len(headings)
    end = max(end, 40)  # cok erken yakalamada da anlamli pencere kalsin
    h = np.unwrap(np.radians(headings[:end]))
    heading_rate = np.degrees(np.diff(h)) / DT_CTRL
    return {
        'heading_rate_rms': float(np.sqrt(np.mean(heading_rate[20:] ** 2))),
        'stick_activity': float(np.mean(np.abs(np.diff(sticks[20:end])))),
        'aim_jump_p95': float(np.percentile(aim_jumps[:end], 95)),
        'capture_t': capture_t if capture_t is not None else float('inf'),
        'mean_dist_last15s': float(np.mean(dists[-int(15.0 / DT_CTRL):])),
        'path': np.array(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true',
                        help='Yol karsilastirma grafigini PNG olarak kaydet')
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()

    modes = ['baseline', 'gate', 'full']
    labels = {'baseline': 'A) Eski kod (zamanlayici update)',
              'gate': 'B) + Olcum kapisi',
              'full': 'C) + Yumusatma + PD yaw + adaptif lead'}
    metrics = ['heading_rate_rms', 'stick_activity', 'aim_jump_p95',
               'capture_t', 'mean_dist_last15s']
    units = {'heading_rate_rms': 'deg/s', 'stick_activity': '',
             'aim_jump_p95': 'cm/s', 'capture_t': 's',
             'mean_dist_last15s': 'cm'}

    results = {m: [run_once(s, m) for s in range(args.seeds)] for m in modes}

    print(f"\nCanli-kosul emulasyonu — {args.seeds} tohum ortalamasi "
          f"(dusuk = iyi, capture_t haric hepsi dalgalanma gostergesi)\n")
    header = f"{'Metrik':<28}" + ''.join(f"{labels[m]:>42}" for m in modes)
    print(header)
    print('-' * len(header))
    for met in metrics:
        row = f"{met + (' [' + units[met] + ']' if units[met] else ''):<28}"
        for m in modes:
            vals = [r[met] for r in results[m]]
            finite = [v for v in vals if math.isfinite(v)]
            if finite:
                cell = f"{np.mean(finite):9.1f} ± {np.std(finite):7.1f}"
                if len(finite) < len(vals):
                    cell += f" ({len(vals) - len(finite)} kacirdi)"
            else:
                cell = 'yakalayamadi'
            row += f"{cell:>42}"
        print(row)

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
        tt = np.array([target_true(t)[:2] for t in np.arange(0, T_END, 0.1)])
        colors = {'baseline': '#ff5252', 'gate': '#ffab40', 'full': '#00e676'}
        for ax, m in zip(axes, modes):
            ax.plot(tt[:, 0], tt[:, 1], color='#29b6f6', lw=1.2, label='Hedef (gercek)')
            ax.plot(results[m][0]['path'][:, 0], results[m][0]['path'][:, 1],
                    color=colors[m], lw=1.5, label='Dron yolu (seed 0)')
            ax.set_title(labels[m], fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle('Canli-Kosul Emulasyonu: Yol Karsilastirmasi')
        out = 'pursuit_smoothing_karsilastirma.png'
        fig.savefig(out, dpi=110, bbox_inches='tight',
                    facecolor='white')
        print(f"\nGrafik kaydedildi: {out}")


if __name__ == '__main__':
    main()
