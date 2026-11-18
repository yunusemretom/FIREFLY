"""
GPS-ONLY HEDEF KESTIRICI — yarisma telemetrisiyle uyumlu
=========================================================
Yarisma sartnamesi hedef icin SADECE enlem, boylam, irtifa ve (yer) hizi verir,
en fazla 5 Hz. Yonelim (rotation) kanali YOKTUR. Bu kestirici yalnizca konum
olcumlerini kullanir (hiz kanali istege bagli).

Model (sabit hizli koordineli donus, yatayda; dikeyde sabit dikey hiz):
    s: referans ana (en son olcumun GERCEK zamani) gore saniye
    psi(s) = psi0 + w*s
    x(s) = x0 + V/w * (sin(psi0 + w s) - sin psi0)
    y(s) = y0 + V/w * (cos psi0 - cos(psi0 + w s))
    z(s) = z0 + vz * s
  Ileri tahminde (s > 0) donus hizi `turn_decay_s` zaman sabitiyle sifira soner.

Olcum: alinis zamani t_k olan konum, hedefin t_k - D anindaki konumudur.
D (veri gecikmesi; sim'de 1.0 s) konum verisinden tek basina gozlemlenemez
(sabit gecikme = iz boyunca geride olmak) -> parametre; kamera ile kalibre
edilebilir (set_delay).

SICRAMA / OFFSET — segment modeli:
  Iki ardisik olcum arasinda fiziksel olarak imkansiz bir adim (hareket
  modelinden `step_cm` fazla sapma) yeni bir SEGMENT baslatir. Her segmentin
  kendi sabit offset'i vardir; oturtmada offset'ler ayrica cozulur, boylece
  sicrama sirasindaki noktalar da dönüs/hiz bilgisini tasimaya devam eder.
  Yeni segmentin ilk noktasi eski bir segmentin seviyesine uyuyorsa (sicrama
  bitti) o segmente katilir. Referans cerceve = son `main_window_s` icinde en
  cok noktaya sahip segment (ana segment); kalici offset degisimi zamanla ana
  segment olur.

Arayuz TrailFollowController ile uyumludur: position(), velocity(), predict(),
trail(lag), V, ready, state(), extra_offset.
"""

import math
from collections import deque

import numpy as np


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _arc(params, s):
    """Yatay model: params=(x0,y0,V,psi0,w), s dizisi -> (N,2)."""
    x0, y0, V, psi0, w = params
    s = np.asarray(s, dtype=float)
    if abs(w) < 1e-5:
        c, sn = math.cos(psi0), math.sin(psi0)
        x = x0 + V * (c * s - 0.5 * w * sn * s * s)
        y = y0 + V * (sn * s + 0.5 * w * c * s * s)
    else:
        a = psi0 + w * s
        x = x0 + V / w * (np.sin(a) - math.sin(psi0))
        y = y0 + V / w * (math.cos(psi0) - np.cos(a))
    return np.column_stack([x, y])


class GpsTargetEstimator:
    def __init__(self, delay_s=1.0, window_s=4.0, min_points=6,
                 speed_bounds=(800.0, 3500.0), max_turn_rate_deg=40.0,
                 outlier_cm=800.0, step_cm=1000.0, merge_cm=1000.0,
                 main_window_s=5.0, turn_decay_s=0.8,
                 use_speed_channel=False, speed_weight=0.2,
                 history_s=30.0, gn_iters=6):
        self.delay = float(delay_s)
        self.window_s = window_s
        self.min_points = min_points
        self.speed_bounds = speed_bounds
        self.max_w = math.radians(max_turn_rate_deg)
        self.outlier_cm = outlier_cm
        self.step_cm = step_cm
        self.merge_cm = merge_cm
        self.main_window_s = main_window_s
        self.turn_decay_s = turn_decay_s
        self.use_speed_channel = use_speed_channel
        self.speed_weight = speed_weight
        self.history_s = history_s
        self.gn_iters = gn_iters

        self._meas = deque()          # (t_alinis, p(3), hiz, segment)
        self._seg_off = {}            # segment -> offset (3) ana cerceveye gore
        self._next_seg = 0
        self.main_seg = None
        self.last_meas_rejected = False   # son olcum ana segment disinda mi
        self.n_segments_started = 0

        self.params = None            # (x0,y0,V,psi0,w) ana cercevede
        self.z0 = 0.0
        self.vz = 0.0
        self._t_ref = None
        self.rms_residual = float("nan")
        self.n_inliers = 0
        self._now = None
        self._trail = deque()         # (gercek zaman, konum) — trail() icin
        self.extra_offset = np.zeros(3)   # disaridan (kamera) duzeltme

    # ------------------------------------------------------------------
    def set_delay(self, d):
        self.delay = float(d)

    def tick(self, t):
        """Olcum gelmese de 'simdi'yi ilerletir (alinis saati)."""
        self._now = t

    def add_position(self, t, pos, speed=None):
        """Hedef konum alani (her tick cagrilabilir; sadece DEGISINCE islenir)."""
        self._now = t if self._now is None else max(self._now, t)
        p = np.asarray(pos, dtype=float)
        if not np.all(np.isfinite(p)) or np.allclose(p, 0.0):
            return False
        if self._meas and np.max(np.abs(p - self._meas[-1][1])) < 1e-6:
            return False
        seg = self._assign_segment(t, p)
        spd = float(speed) if speed is not None and np.isfinite(speed) else float("nan")
        self._meas.append((t, p, spd, seg))
        horizon = max(self.main_window_s, 3 * self.window_s)
        while self._meas and t - self._meas[0][0] > horizon:
            self._meas.popleft()
        live = {m[3] for m in self._meas}
        self._seg_off = {k: o for k, o in self._seg_off.items() if k in live}
        self._update_main()
        self.last_meas_rejected = seg != self.main_seg
        self._refit(t)
        return True

    # ------------------------------------------------------------------
    def _new_seg(self, off):
        k = self._next_seg
        self._next_seg += 1
        self._seg_off[k] = np.asarray(off, dtype=float)
        self.n_segments_started += 1
        return k

    def _assign_segment(self, t, p):
        if not self._meas:
            return self._new_seg(np.zeros(3))
        t0, p0, _, seg0 = self._meas[-1]
        dt = t - t0
        dp = p - p0
        if self.ready and dt < 0.6:
            pred = self._model(np.array([t - self.delay, t0 - self.delay]))
            jump = np.linalg.norm(dp - (pred[0] - pred[1])) > self.step_cm
        else:
            # Kesinti sonrasi: yon bagimsiz kontrol (donus belirsizligine dayanikli)
            V = self.V if self.ready else self.speed_bounds[1]
            jump = abs(np.linalg.norm(dp) - V * dt) > self.step_cm + 300.0 * dt
            if not self.ready:
                jump = np.linalg.norm(dp) > self.speed_bounds[1] * dt + self.step_cm
        if not jump:
            return seg0
        if self.ready:
            base = self._model(np.array([t - self.delay]))[0]
            best, bd = None, self.merge_cm
            for k, o in self._seg_off.items():
                if k == seg0:
                    continue
                d = np.linalg.norm(p - (base + o))
                if d < bd:
                    best, bd = k, d
            if best is not None:
                return best
            return self._new_seg(p - base)
        return self._new_seg(np.zeros(3))

    def _update_main(self):
        t_last = self._meas[-1][0]
        cnt = {}
        for m in self._meas:
            if m[0] > t_last - self.main_window_s:
                cnt[m[3]] = cnt.get(m[3], 0) + 1
        new_main = max(cnt, key=lambda k: (cnt[k], k == self.main_seg))
        if self.main_seg is not None and new_main != self.main_seg and new_main in self._seg_off:
            # Cerceve degisimi: yeni ana segment offset'ini modele aktar
            o = self._seg_off[new_main].copy()
            if self.params is not None:
                self.params = self.params.copy()
                self.params[0] += o[0]
                self.params[1] += o[1]
                self.z0 += o[2]
                self._trail = deque((tt, pp + o) for tt, pp in self._trail)
            for k in self._seg_off:
                self._seg_off[k] = self._seg_off[k] - o
        self.main_seg = new_main
        self._seg_off.setdefault(new_main, np.zeros(3))
        self._seg_off[new_main] = np.zeros(3)

    # ------------------------------------------------------------------
    def _model(self, t_true):
        """Gercek zaman dizisi -> (N,3) konum (ana cerceve, s>0 icin sonumlu donus)."""
        s = np.atleast_1d(np.asarray(t_true, dtype=float) - self._t_ref)
        xy = _arc(self.params, s)
        if self.turn_decay_s > 0 and np.any(s > 0):
            for i in np.nonzero(s > 0)[0]:
                xy[i] = self._extrap_xy(s[i])
        z = self.z0 + self.vz * s
        return np.column_stack([xy, z])

    def _extrap_xy(self, s):
        """psi(s) = psi0 + w*tau*(1-exp(-s/tau)) ile sayisal integral."""
        x0, y0, V, psi0, w = self.params
        tau = self.turn_decay_s
        n = max(4, int(math.ceil(s / 0.05)))
        ss = np.linspace(0.0, s, n + 1)
        psi = psi0 + w * tau * (1.0 - np.exp(-ss / tau))
        c, sn = np.cos(psi), np.sin(psi)
        h = s / n
        x = x0 + V * h * (c.sum() - 0.5 * (c[0] + c[-1]))
        y = y0 + V * h * (sn.sum() - 0.5 * (sn[0] + sn[-1]))
        return np.array([x, y])

    def _init_params(self, s, Z):
        A = np.column_stack([np.ones_like(s), s])
        cx = np.linalg.lstsq(A, Z[:, 0], rcond=None)[0]
        cy = np.linalg.lstsq(A, Z[:, 1], rcond=None)[0]
        V = float(np.clip(math.hypot(cx[1], cy[1]), *self.speed_bounds))
        return np.array([cx[0], cy[0], V, math.atan2(cy[1], cx[1]), 0.0])

    def _gauss_newton(self, x, s, Z, w, spd):
        eps = np.array([1.0, 1.0, 1.0, 1e-4, 1e-5])
        W = np.repeat(w, 2)
        for _ in range(self.gn_iters):
            base = _arc(x, s)
            r = (Z - base).ravel()
            J = np.empty((r.size, 5))
            for k in range(5):
                dx = np.zeros(5)
                dx[k] = eps[k]
                J[:, k] = ((_arc(x + dx, s) - base) / eps[k]).ravel()
            H = J.T @ (J * W[:, None])
            g = J.T @ (W * r)
            if spd is not None:
                ws = self.speed_weight * len(s)
                H[2, 2] += ws
                g[2] += ws * (spd - x[2])
            H += np.diag([1e-6, 1e-6, 1e-6, 1e-2, 1e2])
            try:
                dx = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            x = x + dx
            x[2] = float(np.clip(x[2], *self.speed_bounds))
            x[3] = _wrap(x[3])
            x[4] = float(np.clip(x[4], -self.max_w, self.max_w))
            if np.all(np.abs(dx[:3]) < 0.5) and abs(dx[3]) < 1e-4:
                break
        return x

    def _fit_candidate(self, x, s, Z, seg, spd):
        """Offset'ler ile donusumlu robust GN. -> (x, offsets, w, maliyet)"""
        segs = np.unique(seg)
        off = {k: self._seg_off.get(k, np.zeros(3))[:2].copy() for k in segs}
        off[self.main_seg] = np.zeros(2)
        w = np.ones(len(s))
        for _ in range(4):
            O = np.array([off[k] for k in seg])
            x = self._gauss_newton(x, s, Z[:, :2] - O, w, spd)
            res = Z[:, :2] - O - _arc(x, s)
            # Ana olmayan segment offset'leri: kalintinin medyani
            for k in segs:
                if k == self.main_seg:
                    continue
                m = seg == k
                off[k] = off[k] + np.median(res[m], axis=0)
            O = np.array([off[k] for k in seg])
            rn = np.linalg.norm(Z[:, :2] - O - _arc(x, s), axis=1)
            new_w = (rn < self.outlier_cm).astype(float)
            if new_w.sum() < self.min_points:
                break
            if np.array_equal(new_w, w):
                w = new_w
                break
            w = new_w
        O = np.array([off[k] for k in seg])
        rn = np.linalg.norm(Z[:, :2] - O - _arc(x, s), axis=1)
        cost = float(np.mean(np.minimum(rn, self.outlier_cm) ** 2))
        return x, off, w, cost

    def _refit(self, t):
        good = [m for m in self._meas if m[0] > t - self.window_s]
        if len(good) < self.min_points:
            return
        T = np.array([m[0] for m in good]) - self.delay
        Z = np.array([m[1] for m in good])
        seg = np.array([m[3] for m in good])
        main = seg == self.main_seg
        if main.sum() < 3:
            return
        t_ref = T[-1]
        s = T - t_ref
        spd = None
        if self.use_speed_channel:
            sp = np.array([m[2] for m in good])
            sp = sp[np.isfinite(sp) & (sp > 0)]
            if len(sp):
                spd = float(np.median(sp))

        cands = []
        if self.params is not None:
            cands.append(self._shift(self.params, t_ref - self._t_ref))
        cands.append(self._init_params(s[main], Z[main]))
        best = None
        for x0 in cands:
            r = self._fit_candidate(x0, s, Z, seg, spd)
            if best is None or r[3] < best[3]:
                best = r
        x, off, w, _ = best

        # Dikey: segment offset'li agirlikli dogrusal
        oz = {k: float(self._seg_off.get(k, np.zeros(3))[2]) for k in np.unique(seg)}
        oz[self.main_seg] = 0.0
        vz, z0 = self.vz, self.z0
        for _ in range(2):
            Oz = np.array([oz[k] for k in seg])
            zz = Z[:, 2] - Oz
            ws = max(w.sum(), 1e-9)
            sm = (w * s).sum() / ws
            zm = (w * zz).sum() / ws
            den = (w * (s - sm) ** 2).sum()
            vz = (w * (s - sm) * (zz - zm)).sum() / den if den > 1e-6 else 0.0
            z0 = zm - vz * sm
            for k in oz:
                if k != self.main_seg:
                    m = seg == k
                    oz[k] += float(np.median(Z[m, 2] - oz[k] - (z0 + vz * s[m])))

        self.params, self._t_ref = x, t_ref
        self.vz, self.z0 = float(vz), float(z0)
        for k in off:
            if k != self.main_seg:
                self._seg_off[k] = np.array([off[k][0], off[k][1], oz[k]])
        inl = w > 0
        O3 = np.array([self._seg_off.get(k, np.zeros(3)) for k in seg])
        res = np.linalg.norm(Z - O3 - self._model(T), axis=1)
        self.n_inliers = int(inl.sum())
        self.rms_residual = float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else float("nan")
        self._trail.append((t_ref, self._model(np.array([t_ref]))[0]))
        while self._trail and t_ref - self._trail[0][0] > self.history_s:
            self._trail.popleft()

    @staticmethod
    def _shift(p, ds):
        x0, y0, V, psi0, w = p
        xy = _arc(p, np.array([ds]))[0]
        return np.array([xy[0], xy[1], V, _wrap(psi0 + w * ds), w])

    # ------------------------------------------------------------------
    @property
    def ready(self):
        return self.params is not None

    @property
    def V(self):
        return float(self.params[2]) if self.params is not None else 1790.0

    def _w_at(self, t_true):
        """Donus hizi; son olcumden sonra (s>0) sonumlu."""
        s = t_true - self._t_ref
        w = self.params[4]
        if s > 0 and self.turn_decay_s > 0:
            w *= math.exp(-s / self.turn_decay_s)
        return w

    def heading(self, t_true=None):
        """Model ile tutarli yon (s>0 icin sonumlu donus)."""
        t_true = self._now if t_true is None else t_true
        s = t_true - self._t_ref
        w = self.params[4]
        if s > 0 and self.turn_decay_s > 0:
            return _wrap(self.params[3] + w * self.turn_decay_s * (1.0 - math.exp(-s / self.turn_decay_s)))
        return _wrap(self.params[3] + w * s)

    def position(self):
        return self._model(np.array([self._now]))[0] + self.extra_offset

    def raw_position(self):
        return self._model(np.array([self._now]))[0]

    def velocity(self, t_true=None):
        t_true = self._now if t_true is None else t_true
        psi = self.heading(t_true)
        V = self.params[2]
        return np.array([V * math.cos(psi), V * math.sin(psi), self.vz])

    def predict(self, lead_s):
        return self._model(np.array([self._now + max(0.0, lead_s)]))[0] + self.extra_offset

    def trail(self, lag_s):
        """Hedefin lag_s saniye onceki konumu, hizi, ivmesi (iz takibi icin)."""
        t_q = self._now - lag_s
        if t_q >= self._t_ref - self.window_s or len(self._trail) < 3:
            p = self._model(np.array([t_q]))[0]
            v = self.velocity(t_q)
            psi = math.atan2(v[1], v[0])
            a = self.params[2] * self._w_at(t_q) * np.array([-math.sin(psi), math.cos(psi), 0.0])
            return p + self.extra_offset, v, a
        tt = np.array([q[0] for q in self._trail])
        PP = np.array([q[1] for q in self._trail])
        p = np.array([np.interp(t_q, tt, PP[:, i]) for i in range(3)])
        dt = 0.2
        p0 = np.array([np.interp(t_q - dt, tt, PP[:, i]) for i in range(3)])
        p2 = np.array([np.interp(t_q + dt, tt, PP[:, i]) for i in range(3)])
        v = (p2 - p0) / (2 * dt)
        a = (p2 - 2 * p + p0) / (dt * dt)
        return p + self.extra_offset, v, a

    def state(self):
        rdy = self.ready
        return {
            "ready": rdy,
            "position": self.position() if rdy else None,
            "velocity": self.velocity() if rdy else None,
            "speed": self.V,
            "heading_deg": math.degrees(self.heading()) if rdy else None,
            "turn_rate_deg_s": math.degrees(self._w_at(self._now)) if rdy else 0.0,
            "delay_s": self.delay,
            "n_inliers": self.n_inliers,
            "rms_residual_cm": self.rms_residual,
            "main_segment": self.main_seg,
            "n_segments": len(self._seg_off),
        }
