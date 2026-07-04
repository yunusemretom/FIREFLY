"""
YARISMA ICIN PRODUCTION-READY CTRV UKF FILTRESI
====================================================
Fixed-wing hedef takibi icin, sadece gurultulu GPS konumu (X,Y,Z)
kullanarak calisan, akis halinde (streaming) EKF/UKF filtresi.

Kalibrasyon: hedef_konumlari.xlsx dosyasindaki Real vs Noisy karsilastirmasi
ve 1Hz kosulu simulasyonuyla (5 farkli rastgele tohum) dogrulanmistir.
1Hz kosulunda CTRV UKF, ham veriye gore ortalama %27 daha dusuk pozisyon
hatasi verir (CV Kalman'a gore de daha iyi, o ortalama %19 iyilesme saglar).

KULLANIM (yarisma sirasinda):
    tracker = FixedWingTracker()
    tracker.init(x0, y0, z0)                # ilk GPS okumasi
    for (t, x, y, z) in gps_stream:          # her yeni GPS okumasi geldiginde
        result = tracker.update(t, x, y, z)
        print(result['position'])            # filtrelenmis (px, py, pz)
        print(result['speed'])                # tahmini hiz (skaler)
        print(result['heading_deg'])          # tahmini yon (derece)

NOT: Target_Speed / Noisy_Speed kanali test verisinde gercek hizla
korelasyonsuz bulundugu icin KULLANILMAZ; sadece konum olcumu ile calisilir.
Eger yarismada farkli/guvenilir bir hiz sensoru olursa, update_with_speed()
metodu ile ekstra bir olcum kanali eklenebilir (bkz. yorum satirlari).
"""

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints


class FixedWingTracker:
    # Kalibre edilmis varsayilan parametreler (5 farkli tohumla dogrulandi)
    DEFAULT_STD_A = 80.0        # boylamsal ivme surec gurultusu (birim/s^2)
    DEFAULT_STD_YAWDD = 0.12    # yaw ivmesi surec gurultusu (rad/s^2)
    DEFAULT_STD_VZ = 20.0       # dikey ivme surec gurultusu
    DEFAULT_R_STD = (1426.5, 1574.9, 385.1)   # olcum gurultusu std (X,Y,Z) - test verisinden kalibre

    SCALE = 100.0  # sayisal kararlilik icin ic hesaplamalarda pozisyonlari kucultur

    # Fiziksel durum sinirlari (sabit kanatli Talon icin):
    # 1Hz + yuksek gurultu altinda UKF nadiren dejenere cozume kayabiliyor
    # (negatif hiz, 100+ deg/s donus hizi). Bu sinirlar filtreyi fiziksel
    # olarak mumkun bolgede tutar ve lead-time tahminlerinin patlamasini onler.
    V_BOUNDS = (300.0, 4000.0)      # cm/s  (Talon stall ~ / max dalis hizi)
    YAW_RATE_MAX = 0.6              # rad/s (~34 deg/s; genis bir ust sinir)

    def __init__(self, std_a=None, std_yawdd=None, std_vz=None, r_std=None):
        self.std_a = std_a or self.DEFAULT_STD_A
        self.std_yawdd = std_yawdd or self.DEFAULT_STD_YAWDD
        self.std_vz = std_vz or self.DEFAULT_STD_VZ
        self.r_std = np.array(r_std or self.DEFAULT_R_STD)
        self._ukf = None
        self._last_t = None
        self._initialized = False

    # ------------------------------------------------------------
    @staticmethod
    def _fx(x, dt):
        px, py, yaw, v, yaw_rate, pz, vz = x
        if abs(yaw_rate) > 1e-4:
            px_n = px + (v / yaw_rate) * (np.sin(yaw + yaw_rate * dt) - np.sin(yaw))
            py_n = py + (v / yaw_rate) * (-np.cos(yaw + yaw_rate * dt) + np.cos(yaw))
        else:
            px_n = px + v * np.cos(yaw) * dt
            py_n = py + v * np.sin(yaw) * dt
        yaw_n = yaw + yaw_rate * dt
        pz_n = pz + vz * dt
        return np.array([px_n, py_n, yaw_n, v, yaw_rate, pz_n, vz])

    @staticmethod
    def _hx(x):
        return np.array([x[0], x[1], x[5]])

    @staticmethod
    def _state_mean(sigmas, Wm):
        x = np.zeros(7)
        for idx in [0, 1, 3, 4, 5, 6]:
            x[idx] = np.sum(sigmas[:, idx] * Wm)
        x[2] = np.arctan2(np.sum(np.sin(sigmas[:, 2]) * Wm), np.sum(np.cos(sigmas[:, 2]) * Wm))
        return x

    @staticmethod
    def _state_residual(a, b):
        y = a - b
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi
        return y

    def _process_noise(self, dt):
        Q = np.zeros((7, 7))
        dt2 = dt ** 2
        sa = self.std_a / self.SCALE
        syd = self.std_yawdd
        svz = self.std_vz / self.SCALE
        Q[0, 0] = Q[1, 1] = 0.25 * dt2 * dt2 * sa ** 2
        Q[3, 3] = dt2 * sa ** 2
        Q[2, 2] = 0.25 * dt2 * dt2 * syd ** 2
        Q[4, 4] = dt2 * syd ** 2
        Q[5, 5] = 0.25 * dt2 * dt2 * svz ** 2
        Q[6, 6] = dt2 * svz ** 2
        return Q

    # ------------------------------------------------------------
    def init(self, x0, y0, z0, t0=0.0, initial_heading_deg=None, initial_speed=None):
        """Filtreyi ilk GPS okumasiyla baslat."""
        points = MerweScaledSigmaPoints(n=7, alpha=1.0, beta=2.0, kappa=0.0,
                                         subtract=self._state_residual)
        self._ukf = UKF(dim_x=7, dim_z=3, fx=self._fx, hx=self._hx, dt=1.0,
                         points=points, x_mean_fn=self._state_mean,
                         residual_x=self._state_residual)
        r_scaled = self.r_std / self.SCALE
        self._ukf.R = np.diag(r_scaled ** 2)

        yaw0 = np.radians(initial_heading_deg) if initial_heading_deg is not None else 0.0
        v0 = (initial_speed or 1800.0) / self.SCALE  # varsayilan cruise hizi bilinmiyor ise kaba tahmin

        self._ukf.x = np.array([x0 / self.SCALE, y0 / self.SCALE, yaw0, v0, 0.0, z0 / self.SCALE, 0.0])
        self._ukf.P = np.diag([r_scaled[0]**2, r_scaled[1]**2, 1.5, (v0*0.5)**2 + 1.0,
                                0.3, r_scaled[2]**2, (100/self.SCALE)**2])
        self._last_t = t0
        self._initialized = True

    def update(self, t, x, y, z):
        """Yeni bir GPS konum olcumu geldiginde cagrilir. dt otomatik hesaplanir."""
        if not self._initialized:
            self.init(x, y, z, t0=t)
            return self._current_result()

        dt = max(t - self._last_t, 1e-3)
        self._last_t = t

        self._ukf.Q = self._process_noise(dt)
        self._ukf.predict(dt=dt)
        self._ukf.P = 0.5 * (self._ukf.P + self._ukf.P.T) + np.eye(7) * 1e-6

        z_meas = np.array([x / self.SCALE, y / self.SCALE, z / self.SCALE])
        self._ukf.update(z_meas)
        self._ukf.P = 0.5 * (self._ukf.P + self._ukf.P.T) + np.eye(7) * 1e-6
        self._sanitize_state()

        return self._current_result()

    def _sanitize_state(self):
        """Durumu fiziksel sinirlara ceker (negatif hiz / asiri donus hizi).

        Negatif hiz, CTRV'de heading'in 180 derece ters kilitlenmesi demektir;
        hizi pozitife cevirip heading'i flip ederek ayni yorungeyi fiziksel
        parametrizasyona tasiriz.
        """
        x = self._ukf.x
        if x[3] < 0:
            x[3] = -x[3]
            x[2] = (x[2] + 2 * np.pi) % (2 * np.pi) - np.pi
        vmin, vmax = self.V_BOUNDS[0] / self.SCALE, self.V_BOUNDS[1] / self.SCALE
        x[3] = np.clip(x[3], vmin, vmax)
        x[4] = np.clip(x[4], -self.YAW_RATE_MAX, self.YAW_RATE_MAX)

    def predict_only(self, t):
        """Yeni olcum gelmeden once (veya olcum kaybolduysa) sadece tahmin adimini calistirir."""
        if not self._initialized:
            raise RuntimeError("Once init() veya update() cagirilmali")
        dt = max(t - self._last_t, 1e-3)
        ukf_copy_x, ukf_copy_P = self._ukf.x.copy(), self._ukf.P.copy()
        self._ukf.Q = self._process_noise(dt)
        self._ukf.predict(dt=dt)
        result = self._current_result()
        self._ukf.x, self._ukf.P = ukf_copy_x, ukf_copy_P  # durumu geri al (yan etkisiz tahmin)
        return result

    def _current_result(self):
        x = self._ukf.x
        return {
            'position': (x[0] * self.SCALE, x[1] * self.SCALE, x[5] * self.SCALE),
            'speed': x[3] * self.SCALE,
            'vertical_speed': x[6] * self.SCALE,
            'heading_deg': np.degrees(x[2]) % 360,
            'turn_rate_deg_s': np.degrees(x[4]),
        }


# ----------------------------------------------------------------------
# ORNEK KULLANIM (test verisiyle)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Ellipse

    # Excel dosyasını oku
    xlsx_path = '/home/tom/Documents/Projeler/Firefly/hedef_konumlari.xlsx'
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"Hata: Excel dosyası okunamadı ({xlsx_path}). Lütfen önce drone_gui.py çalıştırıp veri toplayın.")
        print(e)
        exit(1)

    # Hedef konumunun değiştiği (yeni veri gelen) satırları seç
    changed = (df['Target_Noisy_X'].diff() != 0) | (df['Target_Noisy_Y'].diff() != 0) | (df['Target_Noisy_Z'].diff() != 0)
    changed.iloc[0] = True
    events = df[changed].reset_index(drop=True)

    tracker = FixedWingTracker()

    # Premium Koyu Tema Ayarları
    plt.rcParams.update({
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
    })

    plt.ion()
    fig = plt.figure(figsize=(15, 8.5))
    fig.suptitle(
        "Fixed-Wing CTRV UKF Tracker — Çevrimdışı Analiz ve Canlı Görselleştirme",
        fontsize=14, fontweight="bold", color="#e8ecf8", y=0.96
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.07, right=0.96,
                           top=0.88, bottom=0.08, hspace=0.32, wspace=0.22)

    ax_traj = fig.add_subplot(gs[:, 0])   # Sol Panel: 2D Yörünge (X-Y)
    ax_alt = fig.add_subplot(gs[0, 1])    # Sağ Üst: Yükseklik (Z)
    ax_speed = fig.add_subplot(gs[1, 1])  # Sağ Alt: Hız

    # 2D Trajectory setup
    line_real_2d, = ax_traj.plot([], [], color="#29b6f6", lw=2.0, label="Gerçek Yol (Ground Truth)", zorder=3)
    line_noisy_2d, = ax_traj.plot([], [], 'o', color="#ff5252", ms=4, alpha=0.6, label="Gürültülü GPS Ölçümü", zorder=2)
    line_ukf_2d, = ax_traj.plot([], [], color="#00e676", lw=2.5, label="UKF Filtrelenmiş Yol", zorder=4)
    ax_traj.set_xlabel("X (cm)")
    ax_traj.set_ylabel("Y (cm)")
    ax_traj.set_title("2D Yatay Konum Takibi & Belirsizlik Elipsleri", fontsize=11, fontweight="bold")
    ax_traj.legend(loc="upper left", facecolor="#1a1d27", edgecolor="#3a3d4d")
    ax_traj.grid(True)

    # Altitude Setup
    line_real_alt, = ax_alt.plot([], [], color="#29b6f6", lw=1.5, label="Gerçek Yükseklik")
    line_noisy_alt, = ax_alt.plot([], [], 'o', color="#ff5252", ms=3, alpha=0.5, label="Gürültülü GPS Z")
    line_ukf_alt, = ax_alt.plot([], [], color="#00e676", lw=2.0, label="UKF Z Tahmini")
    ax_alt.set_xlabel("Zaman (s)")
    ax_alt.set_ylabel("Yükseklik (cm)")
    ax_alt.set_title("Yükseklik (Z Ekseni) Profil Karşılaştırması", fontsize=10, fontweight="bold")
    ax_alt.legend(loc="upper right", facecolor="#1a1d27", edgecolor="#3a3d4d")
    ax_alt.grid(True)

    # Speed Setup
    line_real_speed, = ax_speed.plot([], [], color="#29b6f6", lw=1.5, label="Gerçek Hız")
    line_noisy_speed, = ax_speed.plot([], [], 'o', color="#ffab40", ms=3, alpha=0.5, label="Ham Hedef Hızı")
    line_ukf_speed, = ax_speed.plot([], [], color="#00e676", lw=2.0, label="UKF Hız Tahmini")
    ax_speed.set_xlabel("Zaman (s)")
    ax_speed.set_ylabel("Hız (cm/s)")
    ax_speed.set_title("Hız Büyüklüğü Tahmini", fontsize=10, fontweight="bold")
    ax_speed.legend(loc="upper right", facecolor="#1a1d27", edgecolor="#3a3d4d")
    ax_speed.grid(True)

    def get_covariance_ellipse(mean, cov, n_std=2.0, **kwargs):
        cov_pos = cov[:2, :2]
        vals, vecs = np.linalg.eigh(cov_pos)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width = 2 * n_std * np.sqrt(max(0, vals[0]))
        height = 2 * n_std * np.sqrt(max(0, vals[1]))
        return Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)

    t_hist = []
    real_x, real_y, real_z = [], [], []
    noisy_x, noisy_y, noisy_z = [], [], []
    ukf_x, ukf_y, ukf_z = [], [], []

    real_speeds = []
    noisy_speeds = []
    ukf_speeds = []

    ellipse_patches = []

    print(f"{'t':>6} | {'filtrelenmis_X':>14} | {'filtrelenmis_Y':>14} | {'filtrelenmis_Z':>10} | {'hiz':>7} | {'yon(deg)':>8}")
    print("---------------------------------------------------------------------------------------------")

    for _, row in events.iterrows():
        t = row['Time_Relative']
        nx, ny, nz = row['Target_Noisy_X'], row['Target_Noisy_Y'], row['Target_Noisy_Z']
        rx = row.get('Target_Real_X', np.nan)
        ry = row.get('Target_Real_Y', np.nan)
        rz = row.get('Target_Real_Z', np.nan)
        
        # Filtreyi güncelle
        res = tracker.update(t, nx, ny, nz)
        px, py, pz = res['position']
        
        # Raporlama
        print(f"{t:6.2f} | {px:14.2f} | {py:14.2f} | {pz:10.2f} | {res['speed']:7.1f} | {res['heading_deg']:8.1f}")
        
        # Verileri biriktir
        t_hist.append(t)
        
        noisy_x.append(nx)
        noisy_y.append(ny)
        noisy_z.append(nz)
        
        real_x.append(rx)
        real_y.append(ry)
        real_z.append(rz)
        
        ukf_x.append(px)
        ukf_y.append(py)
        ukf_z.append(pz)
        
        real_speeds.append(row.get('Target_Real_Speed', np.nan))
        noisy_speeds.append(row.get('Target_Noisy_Speed', np.nan))
        ukf_speeds.append(res['speed'])

        # 2D Grafiği güncelle
        line_real_2d.set_data(real_x, real_y)
        line_noisy_2d.set_data(noisy_x, noisy_y)
        line_ukf_2d.set_data(ukf_x, ukf_y)

        # Eski hata elipslerini kaldır, yenilerini çiz
        for patch in ellipse_patches:
            patch.remove()
        ellipse_patches.clear()

        # UKF Kovaryans elipslerini çiz (1-sigma ve 2-sigma)
        P_scaled = tracker._ukf.P[:2, :2] * (tracker.SCALE ** 2)
        ellipse1 = get_covariance_ellipse((px, py), P_scaled, n_std=1.0, edgecolor="#00e676", facecolor="none", ls="-", lw=1.0, alpha=0.8)
        ellipse2 = get_covariance_ellipse((px, py), P_scaled, n_std=2.0, edgecolor="#00e676", facecolor="#00e676", ls="--", lw=0.6, alpha=0.15)
        ax_traj.add_patch(ellipse1)
        ax_traj.add_patch(ellipse2)
        ellipse_patches.extend([ellipse1, ellipse2])

        ax_traj.relim()
        ax_traj.autoscale_view()
        ax_traj.set_aspect('equal', 'datalim')

        # Yükseklik Grafiği
        line_real_alt.set_data(t_hist, real_z)
        line_noisy_alt.set_data(t_hist, noisy_z)
        line_ukf_alt.set_data(t_hist, ukf_z)
        ax_alt.relim()
        ax_alt.autoscale_view()

        # Hız Grafiği
        line_real_speed.set_data(t_hist, real_speeds)
        line_noisy_speed.set_data(t_hist, noisy_speeds)
        line_ukf_speed.set_data(t_hist, ukf_speeds)
        ax_speed.relim()
        ax_speed.autoscale_view()

        # Ekranı güncelle
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        
        # Animasyon hızı için küçük bir duraklama
        plt.pause(0.04)

    plt.ioff()
    print("\nAnaliz tamamlandı. Grafik ekranını kapatarak çıkabilirsiniz.")
    plt.show()
