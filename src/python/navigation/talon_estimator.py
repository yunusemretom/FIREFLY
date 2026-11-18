"""
TALON KESTIRICI — Gercek zamanli yonelim + gecikmeli GPS fuzyonu
=================================================================
Simulatorden olculen gercekler (analysis/data/rec1.csv, 150 s, debug truth):

  * Hedef `rotation` kanali ~100 Hz gelir, GECIKMESIZ ve BOZULMASIZDIR:
      yaw  == gercek ucus yonu        (RMS 0.01 deg, gecikme 0 s)
      pitch -> dikey hiz V*sin(pitch) (korelasyon 0.99)
      roll  -> donus hizi g*tan(roll)/V (korelasyon 0.94, koordineli donus)
  * Hedef hizi neredeyse sabittir: 1792 +- 23 cm/s (1 s ortalama).
  * Hedef konum kanali ~5 Hz, D saniye GECIKMELI (olculen D = 1.0 s), sabit
    offset + sicrama + kesinti icerir. Hiz kanali kullanilamaz (korelasyon 0.01).

Yontem (rota ezberleme YOK — sadece son birkac saniyenin sensor verisi):
    Birim yon vektoru  u(t) = [cos th cos psi, cos th sin psi, sin th]
    Yol integrali      S(t) = integral u dt        (yonelim kanalindan, 100 Hz)
    Model              p(t) = b + V * S(t)
    Olcum              z_k  = p(t_k - D) + gurultu  (t_k: alinis zamani)

  * D (gecikme) uzun pencerede (varsayilan 20 s) izgara aramasiyla kestirilir
    (donuslerde gozlemlenebilir; duz ucusta b icine emilir ve tahmini bozmaz).
  * b (3) ve V (1) kisa pencerede (6 s) dogrusal en kucuk kareler ile cozulur;
    buyuk kalintili olcumler (sicrama) iteratif olarak atilir.
  * Kesinti/bayat veri: yeni olcum gelmese de S(t) yonelim kanalindan ilerler,
    yani kestirim "olu hesap" ile tam hizda devam eder.

Ileri tahmin (lead): sabit donus hizi + sabit pitch varsayimiyla yay uzerinde.

Bu dosya bilerek filterpy vb. bagimliliklar kullanmaz; numpy yeterlidir.
"""

import math
from collections import deque

import numpy as np


def _wrap_rad(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class TalonEstimator:
    def __init__(self, nominal_speed=1790.0, speed_bounds=(1000.0, 3000.0),
                 delay_grid=(0.0, 3.0, 0.05), delay_window_s=20.0,
                 fit_window_s=6.0, min_fit_points=6, outlier_cm=800.0,
                 attitude_history_s=40.0, turn_rate_tau=0.3,
                 gate_cm=1000.0, gate_persist_s=2.5):
        self.nominal_speed = nominal_speed
        self.speed_bounds = speed_bounds
        self.delays = np.arange(delay_grid[0], delay_grid[1] + 1e-9, delay_grid[2])
        self.delay_window_s = delay_window_s
        self.fit_window_s = fit_window_s
        self.min_fit_points = min_fit_points
        self.outlier_cm = outlier_cm
        self.attitude_history_s = attitude_history_s
        self.turn_rate_tau = turn_rate_tau

        # Yonelim gecmisi: zaman, S (yol integrali), u, psi
        self._att_t = deque()
        self._att_S = deque()
        self._last_u = None
        self._last_psi = None
        self._last_att_t = None
        self._S = np.zeros(3)
        self._turn_rate = 0.0       # rad/s (yaw turevinden, filtrelenmis)
        self._pitch = 0.0
        self._roll = 0.0

        # Olcumler: (t_alinis, p)
        self._meas = deque()
        self._last_meas = None

        # Kestirim
        self.delay = None
        self.b = None
        self.V = nominal_speed
        self.n_inliers = 0
        self.rms_residual = float("nan")
        self._last_fit_t = None
        self._last_delay_t = None
        # Disaridan (kamera) kestirilen GPS offset duzeltmesi; position/predict'e eklenir
        self.extra_offset = np.zeros(3)
        self.gate_cm = gate_cm
        self.gate_persist_s = gate_persist_s
        self._gate_since = None
        self._n_delay_search = 0
        self.last_meas_rejected = False

    # ------------------------------------------------------------------
    # Girdi
    # ------------------------------------------------------------------
    def add_attitude(self, t, roll_deg, pitch_deg, yaw_deg):
        """Hedef yonelimi (her telemetri tick'inde, ~100 Hz)."""
        psi = math.radians(yaw_deg)
        th = math.radians(pitch_deg)
        u = np.array([math.cos(th) * math.cos(psi),
                      math.cos(th) * math.sin(psi),
                      math.sin(th)])
        if self._last_att_t is not None:
            dt = t - self._last_att_t
            if dt <= 0:
                return
            self._S = self._S + 0.5 * (u + self._last_u) * dt
            rate = _wrap_rad(psi - self._last_psi) / dt
            a = 1.0 - math.exp(-dt / self.turn_rate_tau)
            self._turn_rate += a * (rate - self._turn_rate)
        self._last_u = u
        self._last_psi = psi
        self._last_att_t = t
        self._pitch = th
        self._roll = math.radians(roll_deg)
        self._att_t.append(t)
        self._att_S.append(self._S.copy())
        while self._att_t and t - self._att_t[0] > self.attitude_history_s:
            self._att_t.popleft()
            self._att_S.popleft()

    def add_position(self, t, pos):
        """Hedef GPS alani (her tick cagrilabilir; sadece DEGISINCE islenir).

        Donus: True -> yeni olcum islendi.
        """
        p = np.asarray(pos, dtype=float)
        if not np.all(np.isfinite(p)) or np.allclose(p, 0.0):
            return False
        if self._last_meas is not None and np.max(np.abs(p - self._last_meas)) < 1e-6:
            return False
        self._last_meas = p
        # Inovasyon kapisi: mevcut modele gore beklenen olcumden cok uzaksa
        # (sicrama) isaretle. Uzun sure (gate_persist_s) tutarli sekilde uzak
        # kalirsa yeni rejim (or. offset degisimi) kabul edilir.
        flagged = False
        if self.ready and self.delay is not None and len(self._att_t) > 10:
            self._cache_attitude()
            pred = self.b + self.V * self._S_at(np.array([t - self.delay]))[0]
            if np.linalg.norm(p - pred) > self.gate_cm:
                flagged = True
                if self._gate_since is None:
                    self._gate_since = t
                elif t - self._gate_since > self.gate_persist_s:
                    flagged = False
                    self._gate_since = None
                    self._meas = deque((tm, pm, False) for tm, pm, _ in self._meas
                                       if tm >= t - self.gate_persist_s)
            else:
                self._gate_since = None
        self.last_meas_rejected = flagged
        self._meas.append((t, p, flagged))
        horizon = max(self.delay_window_s, self.fit_window_s) + 1.0
        while self._meas and t - self._meas[0][0] > horizon:
            self._meas.popleft()
        self._refit(t)
        return True

    # ------------------------------------------------------------------
    # Kestirim
    # ------------------------------------------------------------------
    def _cache_attitude(self):
        self._tt = np.fromiter(self._att_t, dtype=float)
        self._SS = np.array(self._att_S)

    def _S_at(self, times):
        """Yol integralini verilen zamanlarda (dogrusal interpolasyon) dondurur."""
        tt, SS = self._tt, self._SS
        times = np.clip(times, tt[0], tt[-1])
        return np.column_stack([np.interp(times, tt, SS[:, i]) for i in range(3)])

    def _solve(self, T, Z, d, w=None):
        """z = b + V*S(t-d)  ->  (b, V, residual_norms) [agirlikli LS].

        Kapali form: merkezlenmis verilerde V = <Sc, Zc> / <Sc, Sc>,
        b = Z_ort - V * S_ort  (4 bilinmeyenli LS'nin tam cozumu).
        """
        S = self._S_at(T - d)
        if w is None:
            w = np.ones(len(T))
        ws = max(w.sum(), 1e-9)
        Sm = (S * w[:, None]).sum(0) / ws
        Zm = (Z * w[:, None]).sum(0) / ws
        Sc = S - Sm
        Zc = Z - Zm
        den = float((w[:, None] * Sc * Sc).sum())
        V = float((w[:, None] * Sc * Zc).sum()) / den if den > 1e-9 else self.V
        lo, hi = self.speed_bounds
        V = float(np.clip(V, lo, hi))
        b = Zm - V * Sm
        res = np.linalg.norm(Z - (b + V * S), axis=1)
        return b, V, res

    def _robust_solve(self, T, Z, d):
        w = np.ones(len(T))
        for _ in range(3):
            b, V, res = self._solve(T, Z, d, w)
            new_w = (res < self.outlier_cm).astype(float)
            if new_w.sum() < self.min_fit_points or np.array_equal(new_w, w):
                break
            w = new_w
        inl = w > 0
        cost = float(np.mean(np.minimum(res, self.outlier_cm) ** 2))
        return b, V, res, inl, cost

    def _refit(self, t):
        if len(self._att_t) < 10:
            return
        good = [m for m in self._meas if not m[2]]
        if len(good) < self.min_fit_points:
            return
        T_all = np.array([m[0] for m in good])
        Z_all = np.array([m[1] for m in good])
        self._cache_attitude()
        att_t0 = self._att_t[0]

        # 1) Gecikme: uzun pencere, izgara arama (0.5 s'de bir guncelle)
        if self._last_delay_t is None or t - self._last_delay_t > 0.5 or self.delay is None:
            sel = (T_all > t - self.delay_window_s)
            if sel.sum() >= 2 * self.min_fit_points:
                # Yakinsadiktan sonra sadece mevcut gecikme civarini tara; her
                # 10 aramada bir tam izgara (gecikme degisirse yakalamak icin).
                self._n_delay_search += 1
                grid = self.delays
                if self.delay is not None and self._n_delay_search % 10 != 0:
                    grid = grid[np.abs(grid - self.delay) <= 0.25]
                costs = []
                for d in grid:
                    ok = sel & (T_all - d > att_t0)
                    if ok.sum() < self.min_fit_points:
                        costs.append(np.inf)
                        continue
                    costs.append(self._robust_solve(T_all[ok], Z_all[ok], d)[4])
                costs = np.array(costs)
                if np.isfinite(costs).any():
                    i = int(np.argmin(costs))
                    # Parabolik alt-izgara iyilestirme
                    d_best = grid[i]
                    if 0 < i < len(costs) - 1 and np.all(np.isfinite(costs[i - 1:i + 2])):
                        c0, c1, c2 = costs[i - 1:i + 2]
                        den = c0 - 2 * c1 + c2
                        if den > 1e-9:
                            d_best += 0.5 * (c0 - c2) / den * (self.delays[1] - self.delays[0])
                    self.delay = d_best if self.delay is None else 0.7 * self.delay + 0.3 * d_best
                    self._last_delay_t = t
        if self.delay is None:
            return

        # 2) b, V: kisa pencere
        sel = (T_all > t - self.fit_window_s) & (T_all - self.delay > att_t0)
        if sel.sum() < self.min_fit_points:
            return
        b, V, res, inl, _ = self._robust_solve(T_all[sel], Z_all[sel], self.delay)
        self.b, self.V = b, V
        self.n_inliers = int(inl.sum())
        self.rms_residual = float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else float("nan")
        self._last_fit_t = t

    # ------------------------------------------------------------------
    # Cikti
    # ------------------------------------------------------------------
    @property
    def ready(self):
        return self.b is not None and self._last_u is not None

    def velocity(self):
        return self.V * self._last_u

    def raw_position(self):
        """Sadece GPS+yonelim kestirimi (kamera duzeltmesi YOK)."""
        return self.b + self.V * self._S

    def position(self):
        """Simdiki (en son yonelim zamanindaki) hedef konumu (+kamera duzeltmesi)."""
        return self.b + self.V * self._S + self.extra_offset

    def predict(self, lead_s):
        """lead_s saniye sonrasi icin konum (sabit donus hizi + pitch)."""
        p = self.position()
        if lead_s <= 0:
            return p
        psi = self._last_psi
        th = self._pitch
        w = self._turn_rate
        V = self.V
        Vh = V * math.cos(th)
        if abs(w) > 1e-4:
            dx = (Vh / w) * (math.sin(psi + w * lead_s) - math.sin(psi))
            dy = (Vh / w) * (-math.cos(psi + w * lead_s) + math.cos(psi))
        else:
            dx = Vh * math.cos(psi) * lead_s
            dy = Vh * math.sin(psi) * lead_s
        dz = V * math.sin(th) * lead_s
        return p + np.array([dx, dy, dz])

    def state(self):
        return {
            "ready": self.ready,
            "position": self.position() if self.ready else None,
            "velocity": self.velocity() if self.ready else None,
            "speed": self.V,
            "heading_deg": math.degrees(self._last_psi) if self._last_psi is not None else None,
            "turn_rate_deg_s": math.degrees(self._turn_rate),
            "roll_deg": math.degrees(self._roll),
            "delay_s": self.delay,
            "n_inliers": self.n_inliers,
            "rms_residual_cm": self.rms_residual,
        }
