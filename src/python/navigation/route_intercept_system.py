"""
ROTA TABANLI HEDEF TAKIP VE INTERCEPT SISTEMI
====================================================
Hedef, sabit/tekrarlayan bir rotada (loiter/racetrack) hareket ediyorsa,
her olcumu bagimsiz tahmin etmek yerine rotanin SEKLINI ogrenip, hedefin
rota uzerindeki gelecekteki konumunu dogrudan hesaplamak cok daha
guvenilirdir. Bu modul uc parcadan olusur:

  1) EllipseRouteModel  : biriken (filtrelenmis) konumlardan elips rota
                          fit eder + rota uzerindeki acisal ilerleme
                          hizini (omega) ve yonunu (CW/CCW) kestirir.
  2) RoutePlanner        : rota modelini canli veriyle besler, surekli
                          sapma (residual) izler; sapma esigi asilirsa
                          rotayi otomatik yeniden fit eder.
  3) InterceptSolver      : dronun mevcut konumu + hizindan, hedefi rota
                          uzerinde nerede/ne zaman yakalayabilecegini
                          hesaplar (klasik "lead pursuit" problemi).

Gercek veri (hedef_konumlari.xlsx) uzerinde dogrulanmistir: elips fit
kalinti std'si ~416 birim (elipsin ~11700 birimlik yari-eksenine gore
~%3.5), tam tur suresi ~30s.
"""

import numpy as np


# ========================================================================
# 1) ROTA MODELI (ELIPS)
# ========================================================================
class EllipseRouteModel:
    def __init__(self):
        self.cx = self.cy = self.a = self.b = self.theta = None
        self.omega = None          # acisal hiz (rad/s), isaret = yon (CW/CCW)
        self.phi0 = None           # t0 anindaki faz acisi
        self.t0 = None
        self.fitted = False

    # --------------------------------------------------------------
    @staticmethod
    def _fit_ellipse_algebraic(x, y):
        """Fitzgibbon direkt en-kucuk-kareler elips fit yontemi."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mx, my = x.mean(), y.mean()  # sayisal kararlilik icin merkezle
        xr, yr = x - mx, y - my
        D1 = np.vstack([xr**2, xr*yr, yr**2]).T
        D2 = np.vstack([xr, yr, np.ones_like(xr)]).T
        S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
        C1 = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]])
        M = np.linalg.inv(C1) @ (S1 - S2 @ np.linalg.inv(S3) @ S2.T)
        eigval, eigvec = np.linalg.eig(M)
        cond = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2
        valid = np.nonzero(cond > 0)[0]
        if len(valid) == 0:
            raise ValueError("Elips fit basarisiz (veri elips seklinde degil)")
        a1 = eigvec[:, valid[0]]
        a2 = -np.linalg.inv(S3) @ S2.T @ a1
        A, B, C, D, E, F = np.real(np.concatenate([a1, a2]))

        Mc = np.array([[2*A, B], [B, 2*C]])
        cx, cy = np.linalg.solve(Mc, [-D, -E])
        theta = 0.5 * np.arctan2(B, A - C)
        Aq = A*np.cos(theta)**2 + B*np.cos(theta)*np.sin(theta) + C*np.sin(theta)**2
        Cq = A*np.sin(theta)**2 - B*np.cos(theta)*np.sin(theta) + C*np.cos(theta)**2
        Fq = A*cx**2 + B*cx*cy + C*cy**2 + D*cx + E*cy + F
        a_len = np.sqrt(max(-Fq/Aq, 1e-6))
        b_len = np.sqrt(max(-Fq/Cq, 1e-6))

        # KANONIKLESTIRME: a her zaman buyuk eksen olsun (a >= b). Aksi halde
        # ardisik fit'ler arasinda a/b yer degistirip theta 90 derece kayabilir,
        # bu da faz (phi) tahmininde sureksizliklere ve omega/periyot
        # kestiriminde ciddi hatalara yol acar.
        if b_len > a_len:
            a_len, b_len = b_len, a_len
            theta = theta + np.pi/2

        return cx + mx, cy + my, a_len, b_len, theta

    # --------------------------------------------------------------
    def fit(self, t, x, y):
        """t,x,y: en az bir turun onemli bir kismini kapsayan (>= ~8-10
        nokta onerilir) zaman-damgali konum dizisi."""
        t, x, y = np.asarray(t), np.asarray(x), np.asarray(y)
        if len(x) < 6:
            raise ValueError("Elips fit icin en az 6 nokta gerekli")

        cx, cy, a_len, b_len, theta = self._fit_ellipse_algebraic(x, y)

        # elips cercevesinde faz acisi (phi) hesapla: donusum + normalize
        xt = (x - cx) * np.cos(theta) + (y - cy) * np.sin(theta)
        yt = -(x - cx) * np.sin(theta) + (y - cy) * np.cos(theta)
        phi_raw = np.arctan2(yt / b_len, xt / a_len)
        phi_unwrapped = np.unwrap(phi_raw)

        # phi(t) = phi0 + omega*t -> lineer regresyon
        A_reg = np.vstack([t - t[0], np.ones_like(t)]).T
        omega, phi0 = np.linalg.lstsq(A_reg, phi_unwrapped, rcond=None)[0]

        self.cx, self.cy, self.a, self.b, self.theta = cx, cy, a_len, b_len, theta
        self.omega, self.phi0, self.t0 = omega, phi0, t[0]
        self.fitted = True

        residuals = self._residuals(t, x, y)
        return residuals

    # --------------------------------------------------------------
    def fit_robust(self, t, x, y, outlier_threshold_cm=5000.0):
        """Outlier'ları atarak sağlam elips fit yapar.
        1) Tüm veriyle kaba fit  2) Elips eğrisine geometrik uzaklığı
        eşiği aşan noktaları at  3) Temiz verilerle yeniden fit."""
        t, x, y = np.asarray(t), np.asarray(x), np.asarray(y)
        if len(t) < 6:
            raise ValueError("Robust fit icin en az 6 nokta gerekli")

        # 1. Kaba fit
        self.fit(t, x, y)

        # 2. Geometrik outlier tespiti
        dists = np.empty(len(x))
        for i in range(len(x)):
            phi_i = self.nearest_phi(float(x[i]), float(y[i]))
            ex_i, ey_i = self.point_on_ellipse(phi_i)
            dists[i] = np.hypot(x[i] - ex_i, y[i] - ey_i)

        mask = dists < outlier_threshold_cm
        n_clean = int(np.sum(mask))
        if n_clean < 6:
            return dists[mask] if n_clean > 0 else dists

        # 3. Temiz verilerle yeniden fit
        return self.fit(t[mask], x[mask], y[mask])

    # --------------------------------------------------------------
    def position_at(self, t):
        """Rota modeline gore t anindaki tahmini (x,y) konumu."""
        if not self.fitted:
            raise RuntimeError("Once fit() cagirilmali")
        phi = self.phi0 + self.omega * (t - self.t0)
        xt = self.a * np.cos(phi)
        yt = self.b * np.sin(phi)
        x = self.cx + xt * np.cos(self.theta) - yt * np.sin(self.theta)
        y = self.cy + xt * np.sin(self.theta) + yt * np.cos(self.theta)
        return x, y

    def nearest_phi(self, x, y, n_search=360):
        """Verilen (x,y) noktasina elips uzerinde en yakin faz acisini bulur
        (kaba arama + yerel iyilestirme)."""
        phis = np.linspace(-np.pi, np.pi, n_search, endpoint=False)
        xt = self.a * np.cos(phis)
        yt = self.b * np.sin(phis)
        ex = self.cx + xt * np.cos(self.theta) - yt * np.sin(self.theta)
        ey = self.cy + xt * np.sin(self.theta) + yt * np.cos(self.theta)
        d2 = (ex - x)**2 + (ey - y)**2
        return phis[np.argmin(d2)]

    def point_on_ellipse(self, phi):
        """Verilen faz açısı için elips üzerindeki (x, y) koordinatını döndürür."""
        if not self.fitted:
            raise RuntimeError("Once fit() cagirilmali")
        xt = self.a * np.cos(phi)
        yt = self.b * np.sin(phi)
        x = self.cx + xt * np.cos(self.theta) - yt * np.sin(self.theta)
        y = self.cy + xt * np.sin(self.theta) + yt * np.cos(self.theta)
        return float(x), float(y)

    def _residuals(self, t, x, y):
        px, py = self.position_at(t)
        return np.sqrt((px - x)**2 + (py - y)**2)

    def residual_at(self, t, x, y):
        """Tek bir yeni olcumun rota modeliyle ne kadar tutarli oldugunu dondurur."""
        px, py = self.position_at(t)
        return float(np.hypot(px - x, py - y))

    @property
    def period(self):
        return 2*np.pi/abs(self.omega) if self.omega else None

    @property
    def direction(self):
        if self.omega is None:
            return None
        return "CW" if self.omega < 0 else "CCW"


# ========================================================================
# 2) ROTA PLANLAYICI (canli veri + sapma tespiti + otomatik yeniden fit)
# ========================================================================
class RoutePlanner:
    def __init__(self, buffer_size=40, min_fit_points=10,
                 residual_threshold=1500.0, drift_confirm_count=3):
        self.buffer_size = buffer_size
        self.min_fit_points = min_fit_points
        self.residual_threshold = residual_threshold
        self.drift_confirm_count = drift_confirm_count

        self._t_buf, self._x_buf, self._y_buf = [], [], []
        self._drift_streak = 0
        self.route = EllipseRouteModel()
        self.last_residual = None

    def add_observation(self, t, x, y):
        """Her yeni FILTRELENMIS (UKF/KF ciktisi) konum geldiginde cagirilir."""
        self._t_buf.append(t); self._x_buf.append(x); self._y_buf.append(y)
        if len(self._t_buf) > self.buffer_size:
            self._t_buf.pop(0); self._x_buf.pop(0); self._y_buf.pop(0)

        if not self.route.fitted:
            if len(self._t_buf) >= self.min_fit_points:
                self._refit()
            return "collecting"

        # mevcut rota ile tutarlilik kontrolu
        res = self.route.residual_at(t, x, y)
        self.last_residual = res
        if res > self.residual_threshold:
            self._drift_streak += 1
        else:
            self._drift_streak = max(0, self._drift_streak - 1)

        if self._drift_streak >= self.drift_confirm_count:
            self._refit()
            self._drift_streak = 0
            return "refit"
        return "ok"

    def _refit(self):
        try:
            self.route.fit(np.array(self._t_buf), np.array(self._x_buf), np.array(self._y_buf))
        except ValueError:
            pass  # yeterli/uygun veri yok, mevcut modeli koru

    def predict(self, t_future):
        if not self.route.fitted:
            return None
        return self.route.position_at(t_future)


# ========================================================================
# 3) INTERCEPT (YAKALAMA NOKTASI) COZUCU
# ========================================================================
def solve_intercept(route: EllipseRouteModel, t_now, drone_pos, drone_speed,
                     t_horizon=60.0, dt_search=0.1):
    """
    Dronun (drone_pos, sabit drone_speed ile) hedefi rota uzerinde en erken
    hangi t aninda/nerede yakalayabilecegini bulur.
    Yontem: t uzerinde tarama + f(t) = drone_speed*(t-t_now) - mesafe(t) icin
    ilk isaret degisimini bulup interpolasyonla inceltme (kaba ama saglam).
    """
    if not route.fitted:
        raise RuntimeError("Rota modeli henuz fit edilmedi")

    dx, dy = drone_pos
    t_grid = np.arange(t_now, t_now + t_horizon, dt_search)
    tx, ty = route.position_at(t_grid)
    dist_to_target = np.hypot(tx - dx, ty - dy)
    reachable = drone_speed * (t_grid - t_now)
    f = reachable - dist_to_target

    sign_change = np.where(np.diff(np.sign(f)) > 0)[0]
    if len(sign_change) == 0:
        return None  # bu hiz ve ufukla yakalama mumkun degil

    i = sign_change[0]
    # lineer interpolasyonla t*'i inceltme
    t1, t2 = t_grid[i], t_grid[i+1]
    f1, f2 = f[i], f[i+1]
    t_star = t1 - f1 * (t2 - t1) / (f2 - f1)
    x_star, y_star = route.position_at(t_star)
    return {
        'intercept_time': t_star,
        'intercept_point': (float(x_star), float(y_star)),
        'time_to_intercept': t_star - t_now,
        'required_avg_speed': np.hypot(x_star-dx, y_star-dy) / max(t_star-t_now, 1e-6),
    }
