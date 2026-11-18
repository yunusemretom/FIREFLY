"""
GORSEL KAYMA KESTIRICISI — kameradan GPS offset'ini olcer
==========================================================
TalonEstimator, GPS'in sabit offset'ini (simulatorde 2-3 m) gozlemleyemez;
carpma icin bu hata kanat yari acikligindan (0.86 m) buyuktur. Kamera ise
hedefe giden GERCEK goris hattini (isin) verir.

Her tespit, offset o icin isina DIK iki yonde bir kisit uretir:
    P (p_est + o - p_dron) = 0,   P = I - r r^T     (r: birim isin)
Yani o, (p_est'i isina tasiyan dik duzeltme) ile olculur. Isin boyunca
bilesen tek bakista gozlemlenemez; dron farkli acilardan baktikca (donuslerde)
o da belirginlesir. Kisitlar zamanla unutan (tau) en kucuk kareler ile
birlestirilir; kucuk bir sifir-onsel (prior) cozumu dengeler.

Her olcum, yakalandigi ANDAKI dron pozu ve kestirici konumuyla eslenir; bu
yuzden dedektorun ~160 ms gecikmesi sonucu bozmaz.
"""
import math
import threading
import time

import numpy as np


class VisionOffsetEstimator:
    def __init__(self, tau_s=3.0, prior_weight=0.05, max_offset_cm=800.0, gate_px=120.0):
        self.tau = tau_s
        self.prior = prior_weight
        self.max_offset = max_offset_cm
        self.gate_px = gate_px
        self.A = np.zeros((3, 3))
        self.b = np.zeros(3)
        self._last_t = None
        self.offset = np.zeros(3)
        self.n_used = 0
        self.last_meas_t = None
        self.last_det = None

    def _decay(self, t):
        if self._last_t is not None:
            k = math.exp(-max(t - self._last_t, 0.0) / self.tau)
            self.A *= k
            self.b *= k
        self._last_t = t

    def add(self, t, ray, drone_pos, p_est, score=1.0):
        """ray: dunya cercevesi birim isin; p_est: kestiricinin (offset'siz) hedef konumu."""
        self._decay(t)
        r = np.asarray(ray, float)
        P = np.eye(3) - np.outer(r, r)
        rel = np.asarray(p_est, float) - np.asarray(drone_pos, float)
        # o'nun saglamasi gereken: P (rel + o) = 0  ->  P o = -P rel
        w = float(score)
        self.A += w * P
        self.b += w * (-P @ rel)
        sol = np.linalg.solve(self.A + self.prior * np.eye(3), self.b)
        n = np.linalg.norm(sol)
        if n > self.max_offset:
            sol *= self.max_offset / n
        self.offset = sol
        self.n_used += 1
        self.last_meas_t = t

    def confidence(self, t):
        """Son tespitin tazeligi ve toplam bilgi miktari (0..1)."""
        if self.last_meas_t is None:
            return 0.0
        fresh = math.exp(-max(t - self.last_meas_t, 0.0) / 1.0)
        info = min(1.0, float(np.trace(self.A)) / 6.0)
        return fresh * info


class VisionThread:
    """Arka planda: kare yakala -> ROI tespit -> offset kestiricisini guncelle.

    Ana dongu `publish(t, drone_pos, drone_rot, p_est)` ile guncel durumu verir;
    is parcacigi kare yakaladigi anda bu durumu kopyalar (eslesme).
    """

    def __init__(self, capture, detector, camera, fusion, log=None):
        self.cap = capture
        self.det = detector
        self.cam = camera
        self.fusion = fusion
        self.log = log
        self._state = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.stats = {"frames": 0, "detections": 0, "rejected": 0, "last_px": None}

    def start(self):
        self.th.start()
        return self

    def stop(self):
        self._stop.set()
        self.th.join(timeout=2.0)

    def publish(self, t, drone_pos, drone_rot, p_est):
        with self._lock:
            self._state = (t, tuple(drone_pos), tuple(drone_rot),
                           None if p_est is None else np.array(p_est, float))

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                st = self._state
            if st is None or st[3] is None:
                time.sleep(0.05)
                continue
            try:
                frame = self.cap.grab()
            except Exception:
                time.sleep(0.2)
                continue
            with self._lock:   # yakalama anina en yakin durum
                st = self._state
            t, dpos, drot, p_est = st
            p_corr = p_est + self.fusion.offset
            u, v, z = self.cam.project(p_corr, dpos, drot)
            self.stats["frames"] += 1
            if u is None or not (-200 < u < self.cam.W + 200 and -200 < v < self.cam.H + 200):
                continue
            dets = self.det.detect(frame, center=(u, v))
            if not dets:
                continue
            # Ongorulen noktaya en yakin, yeterince skorlu tespit
            best = min(dets, key=lambda d: math.hypot(d[0] - u, d[1] - v) - 200 * d[4])
            err_px = math.hypot(best[0] - u, best[1] - v)
            # Kapı: offset kestirimi oturduktan sonra uzak tespitleri reddet
            gate = self.fusion.gate_px if self.fusion.n_used > 5 else 400.0
            if err_px > gate:
                self.stats["rejected"] += 1
                continue
            ray = self.cam.ray(best[0], best[1], drot)
            self.fusion.add(t, ray, dpos, p_est, score=best[4])
            self.stats["detections"] += 1
            self.stats["last_px"] = (best[0], best[1], err_px, best[4])
            if self.log is not None:
                self.log.append({"t": t, "u": best[0], "v": best[1], "w": best[2], "h": best[3],
                                 "score": best[4], "pred_u": u, "pred_v": v,
                                 "ox": self.fusion.offset[0], "oy": self.fusion.offset[1],
                                 "oz": self.fusion.offset[2]})
