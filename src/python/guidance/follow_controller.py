"""
IZ TAKIP GUDUMU (trail-follow) + hiz feed-forward'li kontrol
=============================================================
Hedef: Talon'un KENDI IZI uzerinde, `follow_distance` kadar geride, onunla
ayni hizda uçmak; hedefi gecmemek, etrafinda tur atmamak; burnu (kamerayi)
surekli hedefe dogrultmak.

Neden eski yaklasimlar tur atiyordu: dron (3300 cm/s) hedeften (1790 cm/s)
hizli; sadece burnu hedefe cevirmek kapanma hizini kontrol etmez -> gecer,
doner, tekrar gecer. Burada komut HIZ vektorudur:

    p_ref = hedefin t - d/V anindaki konumu      (iz uzerindeki nokta)
    v_ref = hedefin o andaki hiz vektoru          (feed-forward)
    v_des = v_ref + Kp * (p_ref - p_dron)         (kapanma hizi sinirli)

Hata sifira giderken fazla hiz da sifira gider -> asim yok.

Dron modeli (analysis/sysid.py ile simulatorde olculdu, 2026-11-17):
  * egim [deg] = 60 * stick    (pitch+ -> ileri, roll+ -> burnun 90 deg sagi*)
  * kararlı yatay hiz ~= 151 cm/s * egim[deg], 3320 cm/s'de doyar
  * hiz zaman sabiti tau ~= 2 s (birinci derece)
  * yaw stick 0.5 -> ~190 deg/s; yaw+ aciyi arttirir
  * throttle -> dikey hiz DOGRUSAL DEGIL ve 0 civarinda sureksiz:
        +1:3340  +0.2:720  +0.1:410  0:+95  -0.05:+950(!)  -0.3:+510
        -0.5:+165  -0.8:-360  -1:-700   (cm/s, kararlı durum)
    -> (-0.5, 0) bolgesi kullanilmaz; ters harita asagida.
  (*) "sag" = atan2 cercevesinde (psi - 90 deg) yonu.
"""

import math

import numpy as np


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


class DroneModel:
    TILT_PER_STICK = 60.0       # deg
    SPEED_PER_DEG = 151.0       # cm/s / deg (kararli durum)
    TAU_V = 2.0                 # s
    V_MAX = 3300.0              # cm/s
    YAW_RATE_PER_STICK = 380.0  # deg/s (kucuk stickte dogrusal yaklasim)

    @staticmethod
    def throttle_for_vz(vz):
        """Istenen dikey hiz (cm/s) -> throttle (olculen ters harita)."""
        if vz >= 95.0:
            return float(min(1.0, (vz - 95.0) / 3245.0))
        # Alcalma kolu: -0.5 -> +165, -1.0 -> -700  (dogrusal)
        thr = -0.5 - (165.0 - vz) / 1730.0
        return float(max(-1.0, min(-0.5, thr)))


class TrailFollowController:
    def __init__(self, follow_distance=2000.0, altitude_offset=0.0,
                 kp_pos=0.45, kv=1.2, max_close_speed=3300.0, brake_accel=350.0,
                 max_tilt_deg=35.0, kp_alt=0.8, max_vz=1500.0,
                 kp_yaw=4.0, min_separation=1200.0, repulse_gain=2.5,
                 near_dist=6000.0, pass_clearance=3000.0,
                 attack_lead_s=0.15, attack_min_close=500.0, attack_max_close=900.0,
                 attack_kp_alt=1.5, model=DroneModel):
        self.follow_distance = follow_distance
        self.altitude_offset = altitude_offset
        self.kp_pos = kp_pos
        self.kv = kv
        self.max_close_speed = max_close_speed
        self.brake_accel = brake_accel
        self.max_tilt_deg = max_tilt_deg
        self.kp_alt = kp_alt
        self.max_vz = max_vz
        self.kp_yaw = kp_yaw
        self.min_separation = min_separation
        self.repulse_gain = repulse_gain
        self.near_dist = near_dist
        self.pass_clearance = pass_clearance
        self.attack_lead_s = attack_lead_s
        self.attack_min_close = attack_min_close
        self.attack_max_close = attack_max_close
        self.attack_kp_alt = attack_kp_alt
        self.m = model
        self.last = {}

    # ------------------------------------------------------------------
    def reference(self, est):
        """Iz uzerindeki referans nokta ve hizi (TalonEstimator'dan)."""
        V = max(est.V, 1.0)
        lag = self.follow_distance / V
        if hasattr(est, "trail"):
            return est.trail(lag)
        t_now = est._last_att_t
        T = np.array([t_now - lag])
        est._cache_attitude()
        if T[0] < est._tt[0]:
            # Gecmis yetersiz: hiz yonunde geriye
            p_t = est.position()
            u = est.velocity() / V
            return p_t - self.follow_distance * u, est.velocity(), np.zeros(3)
        S_lag = est._S_at(T)[0]
        p_ref = est.b + est.extra_offset + V * S_lag
        # Iz noktasindaki hiz yonu: S'nin turevi
        dt = 0.1
        S0, S1, S2 = est._S_at(np.array([T[0] - dt, T[0], T[0] + dt]))
        v_ref = V * (S2 - S0) / (2 * dt)
        a_ref = V * (S2 - 2 * S1 + S0) / (dt * dt)
        return p_ref, v_ref, a_ref

    # ------------------------------------------------------------------
    def step(self, est, drone_pos, drone_vel, drone_rot, attack=False):
        """-> (throttle, pitch, roll, yaw) stick komutlari + teshis bilgisi.

        attack=True: SALDIRI — referans hedefin kendisi (+kisa lead), minimum
        kapanma hizi korunur (frenleyip durmaz), carpisma onleme kapali.
        """
        p_d = np.asarray(drone_pos, dtype=float)
        v_d = np.asarray(drone_vel, dtype=float)
        roll_d, pitch_d, yaw_d = drone_rot
        p_t = est.position()

        if attack:
            return self._attack_step(est, p_d, v_d, yaw_d, p_t)

        p_ref, v_ref, a_ref = self.reference(est)
        p_ref = p_ref + np.array([0.0, 0.0, self.altitude_offset])

        # --- Yatay hiz komutu ---
        # v_des = v_ref + kapanma. Kapanma hizi uc sinirin en kucugu:
        #   kp*|e| (yakinda dogrusal), sqrt(2*a_fren*|e|) (referansa goreli
        #   hiz sifirla varilsin -> asim/carpisma yok), close_max.
        e = p_ref[:2] - p_d[:2]
        dist_ref = float(np.linalg.norm(e))
        u_e = e / max(dist_ref, 1e-6)
        close = min(self.kp_pos * dist_ref,
                    math.sqrt(2.0 * self.brake_accel * dist_ref),
                    self.max_close_speed)
        v_des = v_ref[:2] + close * u_e
        mode = "IZ" if dist_ref < self.near_dist else "YAKLASMA"

        # DOLANMA: dron hedefin ONUNDE ve izine yakinsa, referansa (hedefin
        # arkasi) dogrudan gitmek hedefin icinden gecmek demektir. Once hedef
        # izinin yanina, pass_clearance mesafede bir noktaya yonel.
        v_t2 = est.velocity()[:2]
        u_t = v_t2 / max(np.linalg.norm(v_t2), 1e-6)
        rel_t = p_d[:2] - p_t[:2]
        along = float(rel_t @ u_t)
        cross_v = rel_t - along * u_t
        cross = float(np.linalg.norm(cross_v))
        if along > -0.5 * self.follow_distance and cross < self.pass_clearance:
            if cross > 1.0:
                n = cross_v / cross
            else:   # tam izin uzerinde: sola (+90 deg) dolan
                n = np.array([-u_t[1], u_t[0]])
            aim = p_t[:2] + n * self.pass_clearance * 1.3 - u_t * 0.5 * self.follow_distance
            to_aim = aim - p_d[:2]
            v_des = self.m.V_MAX * to_aim / max(np.linalg.norm(to_aim), 1e-6)
            mode = "DOLANMA"

        # Carpisma onleme (3B): hedefe min_separation'dan yakinsa uzaklas.
        rel3 = p_d - p_t
        dist_t = float(np.linalg.norm(rel3))
        if 1e-3 < dist_t < self.min_separation:
            rel = rel3[:2] / max(np.linalg.norm(rel3[:2]), 1e-3)
            v_des += self.repulse_gain * (self.min_separation - dist_t) * rel
            mode = "KACINMA"

        vn = np.linalg.norm(v_des)
        if vn > self.m.V_MAX:
            v_des *= self.m.V_MAX / vn

        # --- Hiz -> egim (dunya cercevesi, derece) ---
        # dv/dt = (k*theta - v)/tau  =>  theta = (v_des + tau*a_cmd)/k
        # a_ref: iz noktasinin merkezcil ivmesi (donuste feed-forward).
        a_cmd = self.kv * (v_des - v_d[:2]) + a_ref[:2]
        tilt = (v_des + self.m.TAU_V * a_cmd) / self.m.SPEED_PER_DEG
        tn = np.linalg.norm(tilt)
        if tn > self.max_tilt_deg:
            tilt *= self.max_tilt_deg / tn

        return self._finish(est, p_d, v_d, yaw_d, p_t, p_ref, v_ref, v_des, tilt, mode)

    def _attack_step(self, est, p_d, v_d, yaw_d, p_t):
        lead = self.attack_lead_s
        p_aim = est.predict(lead) if lead > 0 else p_t
        v_t = est.velocity()
        e = p_aim - p_d
        dist = float(np.linalg.norm(e[:2]))
        u_e = e[:2] / max(dist, 1e-6)
        close = float(np.clip(self.kp_pos * dist, self.attack_min_close, self.attack_max_close))
        v_des = v_t[:2] + close * u_e
        vn = np.linalg.norm(v_des)
        if vn > self.m.V_MAX:
            v_des *= self.m.V_MAX / vn
        a_cmd = self.kv * (v_des - v_d[:2])
        tilt = (v_des + self.m.TAU_V * a_cmd) / self.m.SPEED_PER_DEG
        tn = np.linalg.norm(tilt)
        if tn > self.max_tilt_deg:
            tilt *= self.max_tilt_deg / tn
        return self._finish(est, p_d, v_d, yaw_d, p_t, p_aim, v_t, v_des, tilt, "SALDIRI",
                            kp_alt=self.attack_kp_alt)

    def _finish(self, est, p_d, v_d, yaw_d, p_t, p_ref, v_ref, v_des, tilt, mode, kp_alt=None):
        psi = math.radians(yaw_d)
        f = np.array([math.cos(psi), math.sin(psi)])
        r = np.array([math.sin(psi), -math.cos(psi)])   # roll+ yonu (olculen)
        pitch_cmd = float(tilt @ f) / self.m.TILT_PER_STICK
        roll_cmd = float(tilt @ r) / self.m.TILT_PER_STICK

        v_t = est.velocity()
        vz_des = v_t[2] + (kp_alt or self.kp_alt) * (p_ref[2] - p_d[2])
        vz_des = float(np.clip(vz_des, -self.max_vz, self.max_vz))
        thr_cmd = self.m.throttle_for_vz(vz_des)

        los = p_t[:2] - p_d[:2]
        yaw_des = math.degrees(math.atan2(los[1], los[0]))
        yaw_err = wrap_deg(yaw_des - yaw_d)
        yaw_rate = self.kp_yaw * yaw_err + est.state()["turn_rate_deg_s"]
        yaw_cmd = float(np.clip(yaw_rate / self.m.YAW_RATE_PER_STICK, -1.0, 1.0))

        self.last = {
            "p_ref": p_ref, "v_ref": v_ref, "v_des": v_des, "tilt": tilt,
            "err_ref": float(np.linalg.norm(p_ref - p_d)),
            "dist_target": float(np.linalg.norm(p_t - p_d)),
            "yaw_err": yaw_err, "vz_des": vz_des, "mode": mode,
        }
        return (thr_cmd,
                float(np.clip(pitch_cmd, -1, 1)),
                float(np.clip(roll_cmd, -1, 1)),
                yaw_cmd)
