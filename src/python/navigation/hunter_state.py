"""
AVCI DRON DURUM KESTIRICI — seyrek GPS + yer hizi + bilinen komutlar
=====================================================================
Yarismada avcinin hiz VEKTORU gelmez; sadece GPS konum (3.125-50 Hz), skaler
yer hizi ve durus gelir. Kontrolcu hiz vektorune ihtiyac duyar.

Yatay: Kalman filtresi, durum [px, py, vx, vy]. Tahmin adimi olculen dron
modelini kullanir (angle mode):
    egim_dunya [deg] = 60 * stick (komut), egim 0.15 s gecikmeyle izler
    dv/dt = (k * egim - v) / tau,  k = 151 cm/s/deg, tau = 2 s
Komutlari biz verdigimiz icin egim bilinir; boylece seyrek GPS arasinda hiz
dogru ilerler. Guncelleme: GPS konum (dogrusal) + yer hizi buyuklugu (EKF).
Dikey: [z, vz] sabit hiz modeli, irtifa olcumu ile.
"""

import math

import numpy as np


class HunterStateEstimator:
    def __init__(self, k_speed=151.0, tau_v=2.0, tau_tilt=0.15, tilt_per_stick=60.0,
                 v_max=3320.0, sigma_pos=50.0, sigma_speed=30.0, q_acc=150.0,
                 q_vz=300.0, sigma_alt=50.0):
        self.k = k_speed
        self.tau_v = tau_v
        self.tau_tilt = tau_tilt
        self.tilt_per_stick = tilt_per_stick
        self.v_max = v_max
        self.R_pos = sigma_pos ** 2
        self.R_spd = sigma_speed ** 2
        self.q_acc = q_acc
        self.q_vz = q_vz
        self.R_alt = sigma_alt ** 2
        self.x = None                  # [px, py, vx, vy]
        self.P = None
        self.z = None                  # [z, vz]
        self.Pz = None
        self.tilt = np.zeros(2)        # dunya cercevesi egim (deg)
        self._t = None
        self._cmd = (0.0, 0.0)         # (pitch, roll) stick
        self._yaw = 0.0

    # ------------------------------------------------------------------
    def set_command(self, pitch, roll, yaw_deg):
        """Gonderilen stick komutu + o anki yaw (dunya cercevesine cevirmek icin)."""
        self._cmd = (float(pitch), float(roll))
        self._yaw = float(yaw_deg)

    def _tilt_cmd(self):
        psi = math.radians(self._yaw)
        f = np.array([math.cos(psi), math.sin(psi)])
        r = np.array([math.sin(psi), -math.cos(psi)])
        p, ro = self._cmd
        return self.tilt_per_stick * (p * f + ro * r)

    def predict(self, t):
        if self._t is None or self.x is None:
            self._t = t
            return
        dt = t - self._t
        if dt <= 0:
            return
        self._t = t
        a_t = 1.0 - math.exp(-dt / self.tau_tilt)
        self.tilt += a_t * (self._tilt_cmd() - self.tilt)
        v_ss = self.k * self.tilt
        n = np.linalg.norm(v_ss)
        if n > self.v_max:
            v_ss *= self.v_max / n
        e = math.exp(-dt / self.tau_v)
        # Ayrik model: v' = e v + (1-e) v_ss ; p' = p + tau(1-e) v + (dt - tau(1-e)) v_ss
        c = self.tau_v * (1.0 - e)
        F = np.eye(4)
        F[0, 2] = F[1, 3] = c
        F[2, 2] = F[3, 3] = e
        u = np.concatenate([(dt - c) * v_ss, (1.0 - e) * v_ss])
        self.x = F @ self.x + u
        q = self.q_acc ** 2
        Q = q * np.array([[dt ** 3 / 3, 0, dt ** 2 / 2, 0],
                          [0, dt ** 3 / 3, 0, dt ** 2 / 2],
                          [dt ** 2 / 2, 0, dt, 0],
                          [0, dt ** 2 / 2, 0, dt]])
        self.P = F @ self.P @ F.T + Q
        Fz = np.array([[1.0, dt], [0.0, 1.0]])
        self.z = Fz @ self.z
        qz = self.q_vz ** 2
        self.Pz = Fz @ self.Pz @ Fz.T + qz * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])

    def update_gps(self, t, pos):
        pos = np.asarray(pos, dtype=float)
        if self.x is None:
            self.x = np.array([pos[0], pos[1], 0.0, 0.0])
            self.P = np.diag([self.R_pos, self.R_pos, 1e5, 1e5])
            self.z = np.array([pos[2], 0.0])
            self.Pz = np.diag([self.R_alt, 1e5])
            self._t = t
            return
        self.predict(t)
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        self._kf(H, pos[:2] - H @ self.x, self.R_pos * np.eye(2))
        Hz = np.array([[1.0, 0.0]])
        S = Hz @ self.Pz @ Hz.T + self.R_alt
        K = self.Pz @ Hz.T / S
        self.z = self.z + (K * (pos[2] - self.z[0])).ravel()
        self.Pz = (np.eye(2) - K @ Hz) @ self.Pz

    def update_ground_speed(self, t, speed):
        if self.x is None:
            return
        self.predict(t)
        v = self.x[2:4]
        n = np.linalg.norm(v)
        if n < 50.0:
            return              # yon belirsiz; buyukluk guncellemesi kotu kosullu
        H = np.zeros((1, 4))
        H[0, 2:4] = v / n
        self._kf(H, np.array([speed - n]), np.array([[self.R_spd]]))

    def _kf(self, H, y, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    # ------------------------------------------------------------------
    @property
    def ready(self):
        return self.x is not None

    def position(self, t=None):
        if t is not None:
            self.predict(t)
        return np.array([self.x[0], self.x[1], self.z[0]])

    def velocity(self, t=None):
        if t is not None:
            self.predict(t)
        return np.array([self.x[2], self.x[3], self.z[1]])
