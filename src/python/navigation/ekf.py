
import numpy as np


class EKF:
    def __init__(self,
                 sigma_a=1500.0,          
                 r_pos=2500.0,            
                 chi2_threshold=16.27,   
                 dt_nominal=0.05,        
                 dt_max=1.5):            
        self.sigma_a2 = float(sigma_a) ** 2
        self.chi2_threshold = chi2_threshold
        self.dt_nominal = dt_nominal
        self.dt_max = dt_max

        self.x = np.zeros((6, 1))
        self.P = np.eye(6) * 1.0e6        

        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
        self.R = np.eye(3) * r_pos

        self.initialized = False
        self._init_buf = []            
        self._init_n = 4
        self._fake_t = 0.0              
        self._reject_streak = 0
        self._reject_buf = []          
        self._reject_reset = 4          

    def _F(self, dt):
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        return F

    def _Q(self, dt):

        q11 = dt**4 / 4.0
        q12 = dt**3 / 2.0
        q22 = dt**2
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = q11
            Q[i, i + 3] = q12
            Q[i + 3, i] = q12
            Q[i + 3, i + 3] = q22
        return Q * self.sigma_a2

    def initialize(self, pos, vel=(0.0, 0.0, 0.0)):
        self.x[0, 0], self.x[1, 0], self.x[2, 0] = pos
        self.x[3, 0], self.x[4, 0], self.x[5, 0] = vel

        self.P = np.diag([400.0, 400.0, 400.0, 1.0e6, 1.0e6, 1.0e6])
        self.initialized = True

    def predict(self, dt):
        dt = max(1e-3, min(dt, self.dt_max))     # clamp
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        return self.get_position()

    def update(self, pos, t_now=None):

        if not self.initialized:
            if t_now is None:
                t_now = self._fake_t
                self._fake_t += self.dt_nominal
            if not self._init_buf or pos != self._init_buf[-1][0]:
                self._init_buf.append((pos, t_now))
            if len(self._init_buf) < self._init_n:
                return True
            positions = np.array([b[0] for b in self._init_buf], dtype=float)
            times = np.array([b[1] for b in self._init_buf], dtype=float)
            p0 = np.median(positions, axis=0)
            half = len(positions) // 2
            p_early = np.median(positions[:half], axis=0)
            p_late = np.median(positions[half:], axis=0)
            t_early = np.median(times[:half])
            t_late = np.median(times[half:])


            total_span = times[-1] - times[0]
            min_span = max(0.05, total_span * 0.4)    
            span = max(min_span, t_late - t_early)
            v0 = (p_late - p_early) / span


            speed0 = float(np.linalg.norm(v0))
            VMAX = 5000.0
            if speed0 > VMAX:
                v0 = v0 * (VMAX / speed0)
            self.initialize(tuple(p0), tuple(v0))
            return True

        z = np.array(pos, dtype=float).reshape(3, 1)
        y = z - self.H @ self.x                  # innovation
        S = self.H @ self.P @ self.H.T + self.R

        d2 = float((y.T @ np.linalg.inv(S) @ y).item())
        if d2 > self.chi2_threshold:
            self._reject_streak += 1
            self._reject_buf.append(pos)

            if self._reject_streak >= self._reject_reset:
                arr = np.array(self._reject_buf, dtype=float)
                p0 = np.median(arr, axis=0)
                v_keep = self.get_velocity()      # mevcut hizi koru
                self.initialize(tuple(p0), v_keep)
                self._reject_streak = 0
                self._reject_buf = []
                return True
            return False

        self._reject_streak = 0
        self._reject_buf = []
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6)
        # Joseph formu: sayisal olarak kararli, P simetrik+pozitif kalir
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
        return True

    def step(self, pos, dt=None, t_now=None):

        if dt is None:
            dt = self.dt_nominal
        self.predict(dt)
        if pos is None:
            return self.get_position(), False
        accepted = self.update(pos, t_now=t_now)
        return self.get_position(), accepted

    def get_position(self):
        return (self.x[0, 0], self.x[1, 0], self.x[2, 0])

    def get_velocity(self):
        return (self.x[3, 0], self.x[4, 0], self.x[5, 0])

    def get_speed(self):
        vx, vy, vz = self.get_velocity()
        return float(np.sqrt(vx * vx + vy * vy + vz * vz))

    def position_uncertainty(self):
        return float(np.sqrt(self.P[0, 0] + self.P[1, 1] + self.P[2, 2]))

    def predict_future(self, t_ahead):
        px, py, pz = self.get_position()
        vx, vy, vz = self.get_velocity()
        return (px + vx * t_ahead, py + vy * t_ahead, pz + vz * t_ahead)