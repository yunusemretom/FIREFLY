"""
GPS Konum + Hız Verisi Füzyonu — Extended Kalman Filter (EKF)
================================================================

Durum vektörü: x = [px, py, vx, vy]^T  (konum ve hız, 2D)

Ölçüm kaynakları:
  1) GPS konum (px, py)      -> lineer ölçüm
  2) Hız büyüklüğü (|v|)     -> doğrusal olmayan ölçüm (EKF gerektirir)

Outlier koruması:
  - Mahalanobis mesafesi (Normalized Innovation Squared - NIS) ile
    her ölçüm, tahminle ne kadar tutarlı olduğuna göre test edilir.
  - Eşiği aşan ölçümler ya tamamen reddedilir ya da ağırlığı
    (ölçüm gürültü kovaryansı büyütülerek) düşürülür.

Kullanım senaryosu: 1 Hz GPS + hız verisi (örn. OBD/CAN'dan veya GPS
Doppler hızından). Prediction adımı istenirse daha yüksek frekansta
(örn. hız verisi 5-10 Hz geliyorsa) çalıştırılabilir.
"""

import numpy as np


class GpsSpeedEKF:
    def __init__(self, dt=1.0,
                 process_noise_std=0.5,      # ivme belirsizliği (m/s^2 civarı)
                 gps_noise_std=3.0,          # tipik GPS konum std sapması (m)
                 speed_noise_std=0.5,        # hız sensörü std sapması (m/s)
                 gps_gate_threshold=9.21,    # chi-square, 2 dof, %99 güven
                 speed_gate_threshold=6.63): # chi-square, 1 dof, %99 güven
        self.dt = dt

        # Durum: [px, py, vx, vy]
        self.x = np.zeros(4)
        self.P = np.eye(4) * 100.0  # başlangıç belirsizliği yüksek

        self.gps_gate = gps_gate_threshold
        self.speed_gate = speed_gate_threshold

        # Ölçüm gürültü kovaryansları
        self.R_gps = np.eye(2) * gps_noise_std ** 2
        self.R_speed = np.array([[speed_noise_std ** 2]])

        # Process noise ölçekleyici (sabit hız modeli + ivme belirsizliği)
        self.q_std = process_noise_std

        self.initialized = False

    # ------------------------------------------------------------------
    def initialize(self, px, py, vx=0.0, vy=0.0):
        """Filtreyi ilk GPS ölçümüyle başlat."""
        self.x = np.array([px, py, vx, vy], dtype=float)
        self.P = np.diag([self.R_gps[0, 0], self.R_gps[1, 1], 25.0, 25.0])
        self.initialized = True

    # ------------------------------------------------------------------
    def _F(self, dt):
        """Sabit hız (constant velocity) durum geçiş matrisi."""
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def _Q(self, dt):
        """Discrete white noise acceleration process noise modeli."""
        q = self.q_std ** 2
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        return q * np.array([
            [dt4/4, 0,     dt3/2, 0],
            [0,     dt4/4, 0,     dt3/2],
            [dt3/2, 0,     dt2,   0],
            [0,     dt3/2, 0,     dt2]
        ])

    # ------------------------------------------------------------------
    def predict(self, dt=None):
        """Tahmin adımı: bir sonraki konum/hızı hareket modeliyle öngör."""
        if not self.initialized:
            return
        dt = self.dt if dt is None else dt
        F = self._F(dt)
        Q = self._Q(dt)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    # ------------------------------------------------------------------
    def update_gps(self, px_meas, py_meas, adaptive=True):
        """GPS konum ölçümüyle güncelle. Mahalanobis gating uygular."""
        if not self.initialized:
            self.initialize(px_meas, py_meas)
            return True, 0.0

        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        z = np.array([px_meas, py_meas])
        y = z - H @ self.x                      # innovation
        S = H @ self.P @ H.T + self.R_gps        # innovation kovaryansı
        mahalanobis_sq = y.T @ np.linalg.inv(S) @ y

        if mahalanobis_sq > self.gps_gate:
            if adaptive:
                # Tamamen atmak yerine ölçüm gürültüsünü büyüterek
                # ağırlığını azalt (soft rejection) — sık outlier'da
                # filtreyi tamamen "körleştirmemek" için daha güvenli.
                inflate = mahalanobis_sq / self.gps_gate
                S = H @ self.P @ H.T + self.R_gps * inflate
            else:
                return False, mahalanobis_sq  # ölçümü tamamen reddet

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        return True, mahalanobis_sq

    # ------------------------------------------------------------------
    def update_speed(self, speed_meas):
        """
        Hız büyüklüğü ölçümüyle güncelle (doğrusal olmayan -> EKF).
        h(x) = sqrt(vx^2 + vy^2)
        """
        if not self.initialized:
            return True, 0.0

        vx, vy = self.x[2], self.x[3]
        v_norm = np.sqrt(vx ** 2 + vy ** 2)
        eps = 1e-3
        if v_norm < eps:
            v_norm = eps  # sıfıra bölme koruması

        # Jacobian: dh/dx = [0, 0, vx/v, vy/v]
        H = np.array([[0, 0, vx / v_norm, vy / v_norm]])

        y = np.array([speed_meas - v_norm])
        S = H @ self.P @ H.T + self.R_speed
        mahalanobis_sq = float((y.T @ np.linalg.inv(S) @ y).item())

        if mahalanobis_sq > self.speed_gate:
            # Hız ölçümünü zayıflatarak dahil et
            inflate = mahalanobis_sq / self.speed_gate
            S = H @ self.P @ H.T + self.R_speed * inflate

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(4) - K @ H) @ self.P
        return True, mahalanobis_sq

    # ------------------------------------------------------------------
    def step(self, px_meas, py_meas, speed_meas, dt=None):
        """Tek bir zaman adımında tam predict + iki update döngüsü."""
        self.predict(dt)
        gps_ok, gps_score = self.update_gps(px_meas, py_meas)
        speed_ok, speed_score = self.update_speed(speed_meas)
        return {
            "position": (self.x[0], self.x[1]),
            "velocity": (self.x[2], self.x[3]),
            "speed_estimate": np.hypot(self.x[2], self.x[3]),
            "gps_accepted": gps_ok,
            "gps_mahalanobis": gps_score,
            "speed_mahalanobis": speed_score,
        }


# ----------------------------------------------------------------------
# ÖRNEK KULLANIM
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Ellipse

    np.random.seed(42)

    # Filtre Parametreleri
    ekf = GpsSpeedEKF(
        dt=1.0, 
        process_noise_std=1,   # ivme belirsizliği
        gps_noise_std=50.0,       # GPS standart sapması (metre)
        speed_noise_std=0.5      # Hız standart sapması (m/s)
    )

    # 1. Aşama: Gerçek Hareket Trajektörüsü Üretimi (60 Adım)
    # Sabit ivmeli/hızlı ve dönüşler içeren dinamik bir senaryo
    steps = 60
    true_states = []
    px, py = 0.0, 0.0
    vx, vy = 10.0, 2.0
    dt = 1.0

    for i in range(steps):
        # 20. ve 40. adımlar arasında yumuşak bir dönüş (turn) simüle et
        if 20 <= i < 40:
            angle = 0.08 * (i - 20)
            vx_step = 10.0 * np.cos(angle)
            vy_step = 10.0 * np.sin(angle)
        elif i >= 40:
            vx_step = 10.0 * np.cos(0.08 * 20)
            vy_step = 10.0 * np.sin(0.08 * 20)
        else:
            vx_step = vx
            vy_step = vy

        px += vx_step * dt
        py += vy_step * dt
        true_states.append((px, py, vx_step, vy_step))

    # Kayıt Verileri
    t_hist = []
    true_x, true_y = [], []
    gps_x_list, gps_y_list = [], []
    ekf_x, ekf_y = [], []
    
    true_speed = []
    meas_speed = []
    ekf_speed = []
    
    gps_mahalanobis = []
    speed_mahalanobis = []
    
    outliers_x = []
    outliers_y = []

    # Canlı Grafik için Matplotlib Ayarları (Premium Dark Mode)
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
        "Extended Kalman Filter (EKF) — Konum ve Hız Verisi Füzyonu",
        fontsize=14, fontweight="bold", color="#e8ecf8", y=0.96
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.07, right=0.96,
                           top=0.88, bottom=0.08, hspace=0.35, wspace=0.22)

    ax_traj = fig.add_subplot(gs[:, 0])   # Sol Panel: 2D Yörünge
    ax_speed = fig.add_subplot(gs[0, 1])  # Sağ Üst: Hız Grafiği
    ax_nis = fig.add_subplot(gs[1, 1])    # Sağ Alt: Mahalanobis NIS

    # 2D Trajectory Setup
    line_true, = ax_traj.plot([], [], color="#29b6f6", lw=2, label="Gerçek Yörünge", zorder=3)
    line_gps, = ax_traj.plot([], [], 'o', color="#ff5252", ms=4, alpha=0.6, label="Ham GPS Ölçümü", zorder=2)
    line_ekf, = ax_traj.plot([], [], color="#00e676", lw=2.5, label="EKF Konum Tahmini", zorder=4)
    scatter_outliers = ax_traj.scatter([], [], s=80, facecolors='none', edgecolors='#ff1744', lw=1.8, label="Reddedilen GPS (Outlier)", zorder=5)
    
    ax_traj.set_xlabel("X Konumu (metre)")
    ax_traj.set_ylabel("Y Konumu (metre)")
    ax_traj.set_title("2D Konum Takibi ve Kovaryans Belirsizlik Elipsleri", fontsize=11, fontweight="bold")
    ax_traj.legend(loc="upper left", facecolor="#1a1d27", edgecolor="#3a3d4d", fontsize=9)
    ax_traj.grid(True)

    # Speed Setup
    line_true_speed, = ax_speed.plot([], [], color="#29b6f6", ls="--", lw=1.5, label="Gerçek Hız")
    line_meas_speed, = ax_speed.plot([], [], 'o', color="#ffab40", ms=4, alpha=0.6, label="Ölçülen Hız (Sensör)")
    line_ekf_speed, = ax_speed.plot([], [], color="#00e676", lw=2, label="EKF Hız Tahmini")
    ax_speed.set_xlabel("Zaman (sn)")
    ax_speed.set_ylabel("Hız (m/s)")
    ax_speed.set_title("Hız Büyüklüğü Zaman Serisi", fontsize=10, fontweight="bold")
    ax_speed.legend(loc="upper right", facecolor="#1a1d27", edgecolor="#3a3d4d", fontsize=8)
    ax_speed.grid(True)

    # NIS Setup
    line_gps_nis, = ax_nis.plot([], [], color="#29b6f6", lw=1.5, label="GPS Innovation (NIS)")
    line_speed_nis, = ax_nis.plot([], [], color="#b39ddb", lw=1.5, label="Hız Innovation (NIS)")
    ax_nis.axhline(ekf.gps_gate, color="#ff5252", ls="--", lw=1.2, label=f"GPS Limit ({ekf.gps_gate:.1f})")
    ax_nis.axhline(ekf.speed_gate, color="#ff9100", ls="--", lw=1.2, label=f"Hız Limit ({ekf.speed_gate:.1f})")
    ax_nis.set_xlabel("Zaman (sn)")
    ax_nis.set_ylabel("Normalized Innovation Squared (NIS)")
    ax_nis.set_title("Mahalanobis Mesafesi ile Outlier Tespiti", fontsize=10, fontweight="bold")
    ax_nis.legend(loc="upper right", facecolor="#1a1d27", edgecolor="#3a3d4d", fontsize=8)
    ax_nis.grid(True)

    def get_covariance_ellipse(mean, cov, n_std=2.0, **kwargs):
        """Kovaryans matrisi kullanarak 2D hata elipsi oluşturur."""
        cov_pos = cov[:2, :2]
        vals, vecs = np.linalg.eigh(cov_pos)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width = 2 * n_std * np.sqrt(max(0, vals[0]))
        height = 2 * n_std * np.sqrt(max(0, vals[1]))
        return Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)

    ellipse_patches = []

    print("\nEKF Simülasyonu Başlatılıyor...")
    print("----------------------------------------------------------------------")

    for i, (tx, ty, tvx, tvy) in enumerate(true_states):
        # 1. Normal gürültülü sensör verileri üret
        gps_x = tx + np.random.normal(0, 3.0)
        gps_y = ty + np.random.normal(0, 3.0)
        speed = np.hypot(tvx, tvy) + np.random.normal(0, 0.5)

        # 2. Outlier (sapma) enjeksiyonları
        if i == 12:
            # 12. adımda GPS'e kasıtlı büyük bir sıçrama ekle
            gps_x += 80.0
            gps_y -= 60.0
        elif i == 38:
            # 38. adımda bir başka büyük GPS sapması
            gps_x -= 70.0
            gps_y += 50.0
        
        if i == 25:
            # 25. adımda hız sensörüne büyük bir spike (outlier) ekle
            speed += 25.0

        # 3. EKF adımını çalıştır (predict + updates)
        result = ekf.step(gps_x, gps_y, speed)

        # 4. Verileri kaydet
        t_hist.append(i)
        true_x.append(tx)
        true_y.append(ty)
        gps_x_list.append(gps_x)
        gps_y_list.append(gps_y)
        ekf_x.append(result["position"][0])
        ekf_y.append(result["position"][1])
        
        true_speed.append(np.hypot(tvx, tvy))
        meas_speed.append(speed)
        ekf_speed.append(result["speed_estimate"])
        
        gps_mahalanobis.append(result["gps_mahalanobis"])
        speed_mahalanobis.append(result["speed_mahalanobis"])

        # GPS outlier tespiti
        if result["gps_mahalanobis"] > ekf.gps_gate:
            outliers_x.append(gps_x)
            outliers_y.append(gps_y)

        # Terminale rapor et
        flag = ""
        if result["gps_mahalanobis"] > ekf.gps_gate:
            flag += "  <-- GPS OUTLIER (Yumuşatıldı)"
        if result["speed_mahalanobis"] > ekf.speed_gate:
            flag += "  <-- HIZ OUTLIER (Yumuşatıldı)"

        print(f"t={i:2d} | gerçek=({tx:6.1f},{ty:5.1f}) | "
              f"ham_gps=({gps_x:7.1f},{gps_y:6.1f}) | "
              f"filtrelenmis=({result['position'][0]:7.2f},{result['position'][1]:6.2f}) | "
              f"hiz_tahmini={result['speed_estimate']:5.2f} m/s{flag}")

        # 5. Grafikleri Güncelle
        line_true.set_data(true_x, true_y)
        line_gps.set_data(gps_x_list, gps_y_list)
        line_ekf.set_data(ekf_x, ekf_y)
        
        if outliers_x:
            scatter_outliers.set_offsets(np.column_stack((outliers_x, outliers_y)))

        # Eski elipsleri temizle, güncellerini ekle (1-sigma ve 2-sigma)
        for patch in ellipse_patches:
            patch.remove()
        ellipse_patches.clear()

        # 1-sigma (koyu yeşil kenarlık) ve 2-sigma (yarı şeffaf yeşil dolgu)
        ellipse1 = get_covariance_ellipse(result["position"], ekf.P, n_std=1.0, edgecolor="#00e676", facecolor="none", ls="-", lw=1.0, alpha=0.8)
        ellipse2 = get_covariance_ellipse(result["position"], ekf.P, n_std=2.0, edgecolor="#00e676", facecolor="#00e676", ls="--", lw=0.6, alpha=0.15)
        ax_traj.add_patch(ellipse1)
        ax_traj.add_patch(ellipse2)
        ellipse_patches.extend([ellipse1, ellipse2])

        # Eksen limitlerini dinamik ayarla
        ax_traj.relim()
        ax_traj.autoscale_view()
        ax_traj.set_aspect('equal', 'datalim')

        # Hız Grafiği
        line_true_speed.set_data(t_hist, true_speed)
        line_meas_speed.set_data(t_hist, meas_speed)
        line_ekf_speed.set_data(t_hist, ekf_speed)
        ax_speed.relim()
        ax_speed.autoscale_view()

        # NIS Grafiği
        line_gps_nis.set_data(t_hist, gps_mahalanobis)
        line_speed_nis.set_data(t_hist, speed_mahalanobis)
        ax_nis.relim()
        ax_nis.autoscale_view()

        # Ekranı tazele
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        
        # Animasyon hızı için küçük bir duraklama
        plt.pause(0.15)

    plt.ioff()
    print("\nSimülasyon tamamlandı. Grafik penceresini kapatarak çıkabilirsiniz.")
    plt.show()