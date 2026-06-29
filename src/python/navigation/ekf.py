"""
Extended Kalman Filter for fixed-wing target tracking.

State vector: [x, y, z, vx, vy, vz]  (cm and cm/s, Unreal Engine world coords)

Process model: constant velocity.
Speed constraint: after every update |v| is renormalized to constant_speed_cms.
This lets the filter track direction changes (looping/reversing path) while
keeping the speed magnitude anchored to the known value.

Measurement: noisy 3-D position at ~1 Hz with possible:
  - Gaussian noise (FLAG_NOISE)
  - Sudden spikes/jumps (FLAG_JUMP)   → rejected by Mahalanobis gate
  - Frozen dropout (FLAG_DROPOUT)     → detected by comparing consecutive positions
  - Constant offset (FLAG_OFFSET)     → absorbed by filter over time
  - Delayed data (FLAG_DELAY)         → partially handled by covariance growth
"""

import os
import time

import numpy as np
import yaml


class TargetEKF:
    """
    Tracks a fixed-wing target at a known constant speed.

    Typical usage
    -------------
    ekf = TargetEKF()                           # loads config/ekf_params.yaml
    while True:
        raw = drone.get_target_location()       # (x, y, z) in cm
        est = ekf.update(raw)                   # (x, y, z) filtered estimate
        lead = ekf.predict(lookahead_s=0.5)    # where it will be in 0.5 s
    """

    def __init__(self, config_path: str | None = None):
        cfg = self._load_config(config_path)
        t = cfg.get("target_ekf", {})

        self.speed_cms: float = float(t.get("constant_speed_cms", 1500.0))

        pn = t.get("process_noise", {})
        self._q_pos: float = float(pn.get("position", 500.0))
        self._q_vel: float = float(pn.get("velocity", 80_000.0))

        mn = t.get("measurement_noise", {})
        self._r_pos: float = float(mn.get("position", 10_000.0))

        ic = t.get("initial_covariance", {})
        self._p0_pos: float = float(ic.get("position", 1_000_000.0))
        self._p0_vel: float = float(ic.get("velocity", 4_000_000.0))

        rej = t.get("outlier_rejection", {})
        self._maha_thresh: float = float(rej.get("mahalanobis_threshold", 5.0))
        self._min_pos_change: float = float(rej.get("min_position_change_cm", 10.0))

        self._startup_n: int = int(t.get("init", {}).get("startup_samples", 2))

        # 6-state vector [x, y, z, vx, vy, vz]
        self._x = np.zeros(6)
        self._P = np.zeros((6, 6))

        self._initialized: bool = False
        self._startup_buf: list[np.ndarray] = []
        self._last_meas: np.ndarray | None = None
        self._last_t: float | None = None

        # Fixed measurement matrix H: observe positions only
        self._H = np.zeros((3, 6))
        self._H[:3, :3] = np.eye(3)
        self._R = np.eye(3) * self._r_pos

        # Diagnostics exposed for monitoring
        self.last_maha: float = 0.0
        self.last_spike_rejected: bool = False
        self.last_dropout: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        measurement: tuple[float, float, float],
        timestamp: float | None = None,
    ) -> tuple[float, float, float]:
        """
        Feed a new position measurement (raw or repeated) into the filter.

        The method internally decides whether to apply a measurement correction
        based on dropout and spike detection.

        Args:
            measurement: (x, y, z) in cm from the SDK
            timestamp:   Unix time; if None, uses time.time()

        Returns:
            Best current position estimate as (x, y, z).
            Before initialization is complete, returns the raw measurement.
        """
        if timestamp is None:
            timestamp = time.time()

        z = np.asarray(measurement, dtype=float)

        # --- Dropout detection (frozen value) ---
        self.last_dropout = self._is_dropout(z)

        # --- Bootstrap phase ---
        if not self._initialized:
            if not self.last_dropout:
                self._startup_buf.append(z.copy())
                self._last_meas = z.copy()
            if len(self._startup_buf) >= self._startup_n:
                self._init_state(self._startup_buf)
                self._last_t = timestamp
            return (float(z[0]), float(z[1]), float(z[2]))

        # --- Predict ---
        dt = _clamp(timestamp - self._last_t, 0.01, 10.0)
        self._last_t = timestamp
        x_pred, P_pred = self._predict(dt)

        # --- Correct (if not dropout) ---
        self.last_spike_rejected = False
        if self.last_dropout:
            self._x = x_pred
            self._P = P_pred
        else:
            y = z - self._H @ x_pred                      # innovation
            S = self._H @ P_pred @ self._H.T + self._R   # innovation covariance
            try:
                S_inv = np.linalg.inv(S)
                maha = float(np.sqrt(max(0.0, y @ S_inv @ y)))
            except np.linalg.LinAlgError:
                maha = float("inf")
            self.last_maha = maha

            if maha > self._maha_thresh:
                # Spike: propagate prediction only
                self._x = x_pred
                self._P = P_pred
                self.last_spike_rejected = True
            else:
                K = P_pred @ self._H.T @ S_inv
                self._x = x_pred + K @ y
                self._P = (np.eye(6) - K @ self._H) @ P_pred

            self._last_meas = z.copy()

        self._enforce_speed_constraint()
        return self.get_estimated_position()

    def predict(self, lookahead_s: float) -> tuple[float, float, float]:
        """
        Predict where the target will be in `lookahead_s` seconds.

        Useful for lead-angle intercept calculations.
        Returns current estimate if the filter is not yet initialized.
        """
        if not self._initialized:
            return self.get_estimated_position()
        F = _build_F(lookahead_s)
        x_fut = F @ self._x
        return (float(x_fut[0]), float(x_fut[1]), float(x_fut[2]))

    def get_estimated_position(self) -> tuple[float, float, float]:
        """Current filtered position estimate (x, y, z) in cm."""
        return (float(self._x[0]), float(self._x[1]), float(self._x[2]))

    def get_estimated_velocity(self) -> tuple[float, float, float]:
        """Current filtered velocity estimate (vx, vy, vz) in cm/s."""
        return (float(self._x[3]), float(self._x[4]), float(self._x[5]))

    def get_covariance(self) -> np.ndarray:
        """Full 6×6 state covariance matrix."""
        return self._P.copy()

    @property
    def is_ready(self) -> bool:
        """True once the filter has collected enough startup samples."""
        return self._initialized

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = _build_F(dt)
        Q = np.diag([
            self._q_pos, self._q_pos, self._q_pos,
            self._q_vel, self._q_vel, self._q_vel,
        ])
        return F @ self._x, F @ self._P @ F.T + Q

    def _enforce_speed_constraint(self) -> None:
        """Renormalize velocity vector to the known constant speed."""
        v = self._x[3:6]
        mag = float(np.linalg.norm(v))
        if mag > 1e-6:
            self._x[3:6] = v / mag * self.speed_cms

    def _is_dropout(self, z: np.ndarray) -> bool:
        if self._last_meas is None:
            return False
        return float(np.linalg.norm(z - self._last_meas)) < self._min_pos_change

    def _init_state(self, samples: list[np.ndarray]) -> None:
        pos = np.mean(samples, axis=0)
        self._x[:3] = pos
        if len(samples) >= 2:
            dp = samples[-1] - samples[0]
            mag = float(np.linalg.norm(dp))
            self._x[3:6] = (dp / mag * self.speed_cms) if mag > 1e-6 else np.array([self.speed_cms, 0.0, 0.0])
        else:
            self._x[3] = self.speed_cms
        self._P = np.diag([
            self._p0_pos, self._p0_pos, self._p0_pos,
            self._p0_vel, self._p0_vel, self._p0_vel,
        ])
        self._initialized = True

    @staticmethod
    def _load_config(config_path: str | None) -> dict:
        if config_path is None:
            here = os.path.dirname(__file__)
            candidates = [
                os.path.join(here, "../../../config/ekf_params.yaml"),
                "config/ekf_params.yaml",
            ]
            for c in candidates:
                if os.path.exists(c):
                    config_path = c
                    break
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        return {}


# ------------------------------------------------------------------
# Module-level helpers (no state, easily testable)
# ------------------------------------------------------------------

def _build_F(dt: float) -> np.ndarray:
    """Constant-velocity state transition matrix."""
    F = np.eye(6)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))
