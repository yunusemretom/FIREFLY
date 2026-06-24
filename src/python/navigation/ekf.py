"""
ekf.py  —  Extended (lineer CV) Kalman Filter

"""

import numpy as np


class EKF:
    def __init__(self, dt=0.05,
                 q_pos=50.0, q_vel=2500.0,
                 r_pos=900.0,
                 chi2_threshold=150.0):

        self.dt = dt
        self.chi2_threshold = chi2_threshold

        self.x = np.zeros((6, 1))
        self.P = np.eye(6) * 1000.0

        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])

        self.R = np.eye(3) * r_pos

        self.initialized = False

    def initialize(self, pos, vel=(0.0, 0.0, 0.0)):
        self.x[0, 0], self.x[1, 0], self.x[2, 0] = pos
        self.x[3, 0], self.x[4, 0], self.x[5, 0] = vel
        self.P = np.eye(6) * 100.0
        self.initialized = True

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_position()

    def update(self, pos):

        if not self.initialized:
            if not hasattr(self, "_last_meas") or self._last_meas is None:
                self.initialize(pos)
                self._last_meas = pos
                return True
            # İkinci ölçüm: iki nokta farkından hızı tahmin et (hızlı yakınsama)
            vel = tuple((p - lp) / self.dt for p, lp in zip(pos, self._last_meas))
            self.initialize(pos, vel)
            self._last_meas = pos
            return True

        z = np.array(pos, dtype=float).reshape(3, 1)

        # Innovation (ölçüm - tahmin)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R

        mahalanobis = float((y.T @ np.linalg.inv(S) @ y).item())
        if mahalanobis > self.chi2_threshold:
            # Ölçüm tutarsız (ani sıçrama) -> reddet, sadece predict geçerli kalsın
            return False

        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return True

    def step(self, pos):
 
        self.predict()
        if pos is None:
            return self.get_position(), False 
        accepted = self.update(pos)
        return self.get_position(), accepted

    def get_position(self):
        return (self.x[0, 0], self.x[1, 0], self.x[2, 0])

    def get_velocity(self):
        return (self.x[3, 0], self.x[4, 0], self.x[5, 0])

    def get_speed(self):
        vx, vy, vz = self.get_velocity()
        return float(np.sqrt(vx**2 + vy**2 + vz**2))

    def predict_future(self, t_ahead):
        px, py, pz = self.get_position()
        vx, vy, vz = self.get_velocity()
        return (px + vx * t_ahead, py + vy * t_ahead, pz + vz * t_ahead)