"""
PURSUIT GUIDANCE v2 — Puruzsuz, Canli-Kosul Uyumlu Takip Rehberligi
====================================================================
Onceki surumun (tests/python/pursuit_guidance (1).py) canli modda dalgali
yol izlemesinin kok nedenleri ve bu moduldeki cozumleri:

1) BAYAT OLCUM (stale measurement) SORUNU:
   drone_sdk telemetrisinde hedef GPS alani, yeni paket gelmese de son
   degeri tutar. Zamanlayici tabanli `gps_counter` ile update atmak,
   ayni olcumu filtreye tekrar besler veya olcumu yanlis zaman damgasiyla
   isler -> UKF "hedef yavasladi/durdu" sanir -> hiz/heading tahmini
   salinir -> lead point ileri-geri sicrar -> dalgali yol.
   COZUM: MeasurementGate — telemetri degeri DEGISMEDIKCE filtreye olcum
   beslenmez (change-detection). Zamanlayici tamamen kaldirildi.

2) OLCUM ANINDA LEAD-POINT SICRAMASI:
   Her update() gurultulu olcume dogru cekilir (R std ~1500 cm). Konum ve
   ozellikle heading tahminindeki ani degisim, lead_time ile carpilarak
   lead noktasini yanal yuzlerce cm kaydirir.
   COZUM: LeadPointSmoother — lead noktasi slew-limit + alcak geciren
   filtre ile yumusatilir. Sicrama tek tick yerine ~tau saniyeye yayilir.

3) YAW KONTROLUNDE FAZ GECIKMESI:
   P-only + agir EMA (0.7/0.3) gecikme ekler; gecikme + P -> LOS etrafinda
   salinim. COZUM: YawController — PD kontrol (turev sonumleme) + hafif
   filtreleme. Turev, yumusatilmis lead noktasindan geldigi icin sakindir.

KULLANIM (canli dongu, 10 Hz):
    tracker  = FixedWingTracker()
    guidance = SmoothPursuitGuidance(tracker, lead_time=1.0)
    yaw_ctrl = YawController()

    while running:
        t_now = time.time() - t0
        tar_pos = telemetry["target"]["position"]

        guidance.ingest(t_now, tar_pos)          # sadece YENI olcumde update
        aim = guidance.compute_aim(t_now)        # yumusatilmis hedef nokta

        yaw_cmd = yaw_ctrl.step(t_now, bearing_to(aim), drone_yaw_deg)
"""

import math
import numpy as np


def wrap_deg(angle):
    """Aciyi (-180, 180] araligina indirger."""
    return (angle + 180.0) % 360.0 - 180.0


class MeasurementGate:
    """Hedef GPS telemetrisinin *gercekten yeni* olup olmadigini tespit eder.

    Telemetri alani stale (bayat) deger tutabildigi icin, deger degismedikce
    olcum yeni sayilmaz. Gurultulu float GPS'te iki ardisyik olcumun birebir
    ayni olma olasiligi ihmal edilebilir.
    """

    def __init__(self, min_change_cm=1e-6):
        self.min_change = min_change_cm
        self._last = None
        self.last_accept_t = None

    def is_new(self, pos):
        if self._last is None:
            self._last = tuple(pos)
            return True
        changed = any(abs(a - b) > self.min_change
                      for a, b in zip(pos, self._last))
        if changed:
            self._last = tuple(pos)
        return changed


class LeadPointSmoother:
    """Lead noktasindaki ani sicramalari yayarak puruzsuz bir hedef uretir.

    - Alcak geciren filtre (tau): olcum ani sicramasini ~tau saniyeye yayar.
    - Slew limiti: yumusatilmis nokta, hedef hizinin `slew_factor` katindan
      hizli hareket edemez. slew_factor > 1 olmali ki nokta hedefi surekli
      takip edebilsin; 2.0 yakinsama icin rahat marj birakir.
    - Slew tabani NOMINAL hedef hizidir (filtre tahmini degil): UKF'nin hiz
      tahmini gecici olarak cokerse bile aim noktasi gercek lead'i takip
      edebilecek hizda kalir.
    - max_lag: aim noktasi ham lead noktasindan bu mesafeden fazla geride
      kalamaz (filtre sicramasi cok buyukse yumusatma yerine sinirli takip).

    Kararli durumda LPF gecikmesi ~ v_hedef * tau geriden takip demektir
    (orn. 1800 cm/s * 0.3 s = 540 cm); lead_time zaten ~1800 cm ileriyi
    gosterdigi icin net hedef hala hedefin onundedir.
    """

    def __init__(self, tau=0.3, slew_factor=2.0, nominal_speed=1500.0,
                 max_lag=3000.0):
        self.tau = tau
        self.slew_factor = slew_factor
        self.nominal_speed = nominal_speed
        self.max_lag = max_lag
        self._point = None
        self._last_t = None

    def reset(self):
        self._point = None
        self._last_t = None

    def filter(self, t, raw_point, target_speed):
        raw = np.asarray(raw_point, dtype=float)
        if self._point is None:
            self._point = raw.copy()
            self._last_t = t
            return tuple(self._point)

        dt = max(t - self._last_t, 1e-3)
        self._last_t = t

        alpha = 1.0 - math.exp(-dt / self.tau)
        step = alpha * (raw - self._point)

        max_step = self.slew_factor * max(abs(target_speed),
                                          self.nominal_speed) * dt
        step_len = np.linalg.norm(step)
        if step_len > max_step:
            step *= max_step / step_len

        self._point += step

        # Guvenlik: aim, ham lead noktasindan max_lag'den fazla geride kalamaz.
        gap = raw - self._point
        gap_len = np.linalg.norm(gap)
        if gap_len > self.max_lag:
            self._point = raw - gap * (self.max_lag / gap_len)

        return tuple(self._point)


class SmoothPursuitGuidance:
    """MeasurementGate + FixedWingTracker + LeadPointSmoother zinciri.

    lead_time: hedefin kac saniye ilerisine yonelinecegi.
    adaptive_lead=True ise donus hizi yukseldikce lead kisaltilir
    (CTRV ekstrapolasyonu viraj sirasinda daha hizli sapar).
    """

    def __init__(self, tracker, lead_time=1.0, min_lead=0.3, max_lead=2.5,
                 smoother_tau=0.3, slew_factor=2.0, adaptive_lead=True):
        self.tracker = tracker
        self.base_lead = float(np.clip(lead_time, min_lead, max_lead))
        self.min_lead = min_lead
        self.max_lead = max_lead
        self.adaptive_lead = adaptive_lead
        self.gate = MeasurementGate()
        self.smoother = LeadPointSmoother(tau=smoother_tau,
                                          slew_factor=slew_factor)
        self.last_result = None

    def ingest(self, t, target_pos):
        """Her kontrol tick'inde cagrilir; olcum sadece degistiyse islenir.

        Donus: True -> filtre yeni olcumle guncellendi, False -> bayat veri.
        """
        if not self.gate.is_new(target_pos):
            return False
        self.last_result = self.tracker.update(t, *target_pos)
        self.gate.last_accept_t = t
        return True

    def current_lead_time(self):
        if not self.adaptive_lead or self.last_result is None:
            return self.base_lead
        # 10 deg/s donuste lead yariya iner; duz ucusta base_lead kullanilir.
        turn = abs(self.last_result.get('turn_rate_deg_s', 0.0))
        lead = self.base_lead / (1.0 + turn / 10.0)
        return float(np.clip(lead, self.min_lead, self.base_lead))

    def compute_aim(self, t):
        """Yumusatilmis nisan noktasi + tracker durum bilgisi dondurur."""
        lead = self.current_lead_time()
        pred = self.tracker.predict_only(t + lead)
        raw_point = pred['position']
        aim = self.smoother.filter(t, raw_point, pred['speed'])
        return {
            'aim_point': aim,
            'raw_lead_point': raw_point,
            'lead_time': lead,
            'target_speed': pred['speed'],
            'target_heading_deg': pred['heading_deg'],
        }


class YawController:
    """P + gyro-sonumlemeli yaw kontrolu.

    Onceki surumdeki agir EMA (0.7 eski + 0.3 yeni) faz gecikmesi ekleyip
    salinim uretiyordu; burada EMA yok. Sonumleme, hata turevinden DEGIL
    dronun kendi olculen yaw hizindan alinir (gyro damping): hedef/aim
    noktasindaki gurultu D terimine sizamaz, sadece dronun gercek donusu
    frenlenir. Cikti [-1, 1] araliginda normalize stick komutudur.

    kd secimi: kd * max_yaw_rate, kp * tipik_hata ile ayni mertebede
    olmali (orn. kd=0.008, 100 deg/s'de 0.8'lik fren).
    """

    def __init__(self, kp=0.07, kd=0.008, rate_filter_tau=0.1):
        self.kp = kp
        self.kd = kd
        self.rate_filter_tau = rate_filter_tau
        self._prev_yaw = None
        self._prev_t = None
        self._rate_filt = 0.0

    def reset(self):
        self._prev_yaw = None
        self._prev_t = None
        self._rate_filt = 0.0

    def step(self, t, target_bearing_deg, current_yaw_deg):
        err = wrap_deg(target_bearing_deg - current_yaw_deg)

        damping = 0.0
        if self._prev_yaw is not None:
            dt = max(t - self._prev_t, 1e-3)
            raw_rate = wrap_deg(current_yaw_deg - self._prev_yaw) / dt
            beta = 1.0 - math.exp(-dt / self.rate_filter_tau)
            self._rate_filt += beta * (raw_rate - self._rate_filt)
            damping = self.kd * self._rate_filt

        self._prev_yaw = current_yaw_deg
        self._prev_t = t

        cmd = self.kp * err - damping
        return float(np.clip(cmd, -1.0, 1.0))
