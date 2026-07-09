"""
ROTA TESTLERI ORTAK GORSELLESTIRME / SIMULASYON YARDIMCILARI
=============================================================
test_route_intercept.py ve test_route_kamikaze.py'nin paylastigi matplotlib
tema, canli harita panosu (dashboard) ve elips ground-truth simulasyon hedefi.
Bu scaffolding onceden iki dosyada satir satir kopyalanmisti.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Premium koyu tema (canlı + sim panolarında ortak)
DARK_THEME = {
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#3a3d4d",
    "axes.labelcolor":   "#c8ccd8",
    "xtick.color":       "#7a7d8d",
    "ytick.color":       "#7a7d8d",
    "text.color":        "#c8ccd8",
    "grid.color":        "#2a2d3d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
}


def apply_dark_theme():
    plt.rcParams.update(DARK_THEME)


class LiveDashboard:
    """3 panelli canlı harita: sol büyük XY haritası, sağ üst mesafe, sağ alt
    elips kalıntısı. Artist'ler set_data ile güncellenir. intercept ve kamikaze
    aynı düzeni sadece başlık/etiket farkıyla kullanır."""

    def __init__(self, suptitle, map_title, target_label="Hedef GPS",
                 intercept_label="Yakalama Noktası (Intercept)"):
        apply_dark_theme()
        plt.ion()
        self.fig = plt.figure(figsize=(15, 8.5))
        self.fig.suptitle(suptitle, fontsize=14, fontweight="bold",
                          color="#e8ecf8", y=0.96)

        gs = gridspec.GridSpec(2, 2, figure=self.fig, left=0.07, right=0.96,
                               top=0.88, bottom=0.08, hspace=0.32, wspace=0.22)
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self.ax_dist = self.fig.add_subplot(gs[0, 1])
        self.ax_res = self.fig.add_subplot(gs[1, 1])

        m = self.ax_map
        self.line_target_path, = m.plot([], [], 'o', color="#ff5252", ms=3, alpha=0.4, label=target_label)
        self.line_fitted_ellipse, = m.plot([], [], color="#ce93d8", lw=2, label="Tahmini Rota (Elips)")
        self.line_drone_path, = m.plot([], [], color="#29b6f6", lw=1.5, label="Dron Rotası")
        self.point_target, = m.plot([], [], 'o', color="#ff1744", ms=8, label="Hedef (Şu anki)")
        self.point_drone, = m.plot([], [], '^', color="#00e5ff", ms=8, label="Dron")
        self.point_intercept, = m.plot([], [], '*', color="#00e676", ms=12, label=intercept_label)
        self.line_intercept_vector, = m.plot([], [], '--', color="#00e676", alpha=0.7)
        m.set_xlabel("X (cm)"); m.set_ylabel("Y (cm)")
        m.set_title(map_title, fontsize=11, fontweight="bold")
        m.legend(loc="upper left", facecolor="#1a1d27", edgecolor="#3a3d4d")
        m.grid(True)

        self.line_dist, = self.ax_dist.plot([], [], color="#ffab40", lw=1.5)
        self.ax_dist.set_xlabel("Zaman (s)"); self.ax_dist.set_ylabel("Mesafe (cm)")
        self.ax_dist.set_title("Dron ve Hedef Arasındaki Mesafe", fontsize=10, fontweight="bold")
        self.ax_dist.grid(True)

        self.line_res, = self.ax_res.plot([], [], color="#b39ddb", lw=1.5)
        self.ax_res.set_xlabel("Zaman (s)"); self.ax_res.set_ylabel("Hata (cm)")
        self.ax_res.set_title("Elips Rota Kalıntı Hatası (Residuals)", fontsize=10, fontweight="bold")
        self.ax_res.grid(True)

    def set_ellipse(self, route):
        """Fit edilmiş elipsin çizgisini günceller."""
        phis = np.linspace(-np.pi, np.pi, 200)
        ex = (route.cx + route.a * np.cos(phis) * np.cos(route.theta)
              - route.b * np.sin(phis) * np.sin(route.theta))
        ey = (route.cy + route.a * np.cos(phis) * np.sin(route.theta)
              + route.b * np.sin(phis) * np.cos(route.theta))
        self.line_fitted_ellipse.set_data(ex, ey)

    def update(self, target_xy_log, drone_xy_log, tx, ty, dx, dy,
               t_hist, dist_log, res_log, intercept_pt):
        self.line_target_path.set_data(target_xy_log[0], target_xy_log[1])
        self.line_drone_path.set_data(drone_xy_log[0], drone_xy_log[1])
        self.point_target.set_data([tx], [ty])
        self.point_drone.set_data([dx], [dy])
        if intercept_pt is not None:
            self.point_intercept.set_data([intercept_pt[0]], [intercept_pt[1]])
            self.line_intercept_vector.set_data([dx, intercept_pt[0]], [dy, intercept_pt[1]])
        else:
            self.point_intercept.set_data([], [])
            self.line_intercept_vector.set_data([], [])

        self.ax_map.relim(); self.ax_map.autoscale_view()
        self.ax_map.set_aspect('equal', 'datalim')
        self.line_dist.set_data(t_hist, dist_log)
        self.ax_dist.relim(); self.ax_dist.autoscale_view()
        self.line_res.set_data(t_hist, res_log)
        self.ax_res.relim(); self.ax_res.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def block(self):
        plt.ioff()
        plt.show()


class EllipseSimTarget:
    """Elips yörüngede sabit açısal hızla uçan ground-truth hedef + GPS gürültüsü.
    Simülasyon modu için (canlı SDK yerine)."""

    def __init__(self, cx, cy, a, b, speed_cms, gps_noise_cm):
        self.cx, self.cy, self.a, self.b = cx, cy, a, b
        self.gps_noise_cm = gps_noise_cm
        self.omega = speed_cms / math.sqrt((a**2 + b**2) / 2.0)

    def truth(self, t):
        return (self.cx + self.a * math.cos(self.omega * t),
                self.cy + self.b * math.sin(self.omega * t))

    def measure(self, t):
        tx, ty = self.truth(t)
        return (tx + np.random.normal(0, self.gps_noise_cm),
                ty + np.random.normal(0, self.gps_noise_cm))
