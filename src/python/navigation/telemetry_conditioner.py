"""
TELEMETRI KOSULLANDIRMA — Olcum Kapisi + Irtifa Kilidi
======================================================
Hedef GPS telemetrisi gercek bir sensor gibi davranir: en fazla 1 Hz gelir,
yeni paket gelmese de SDK son degeri TUTAR (stale hold), gurultu/offset/ani
spike icerir ve gecikmeli olabilir. Bu modul ham telemetriyi gudume
beslemeden once temizleyen iki kucuk durum makinesi saglar:

  * MeasurementGate : konum DEGISMEDIKCE yeni olcum uretmez (change-detection),
                      500 m'yi asan ani sicramalari (bozuk paket) eler,
                      uzun sure yeni olcum gelmezse "bayat" (stale) bayragi kaldirir.
  * AltitudeLock    : hedefin gercek irtifasi degismedigi icin Z'yi, ani
                      sicramalari reddeden bir hareketli ortalamayla kilitler.
                      Es-irtifa kilidi 2B kesisimi 3B carpismaya cevirir.

Her iki sinif da test_route_intercept.py ve test_route_kamikaze.py tarafindan
paylasilir; onceden bu mantik iki dosyada satir satir kopyalanmisti.
"""

import math
from collections import namedtuple

GateResult = namedtuple(
    "GateResult", ["x", "y", "meas_changed", "data_valid", "data_stale"]
)


class MeasurementGate:
    """Ham (x, y) hedef konumunu koşullandırır.

    jump_reject_cm : bu mesafeden büyük ani sıçrama = bozuk paket, elenir.
    stale_timeout_s: bu süre boyunca yeni ölçüm gelmezse data_stale=True.
    """

    def __init__(self, jump_reject_cm=50000.0, stale_timeout_s=2.5):
        self.jump_reject_cm = jump_reject_cm
        self.stale_timeout_s = stale_timeout_s
        self.last_x = None
        self.last_y = None
        self.last_meas_t = None

    def update(self, t_now, x_raw, y_raw):
        # Konum verisi geçerli mi (0,0 = bozuk/henüz akmıyor)
        data_valid = not (x_raw == 0.0 and y_raw == 0.0)

        # Değişim tespiti: değer öncekiyle aynıysa SDK bayat veriyi tutuyordur
        meas_changed = data_valid and (x_raw != self.last_x or y_raw != self.last_y)
        if meas_changed and self.last_x is not None:
            jump = math.hypot(x_raw - self.last_x, y_raw - self.last_y)
            if jump > self.jump_reject_cm:  # ani spike = bozuk
                data_valid = False
                meas_changed = False

        if meas_changed:
            x, y = x_raw, y_raw
            self.last_x, self.last_y = x, y
            self.last_meas_t = t_now
        else:
            # Bayat/bozuk veri — son geçerli konumu kullan
            x = self.last_x if self.last_x is not None else x_raw
            y = self.last_y if self.last_y is not None else y_raw

        data_stale = (self.last_meas_t is not None
                      and (t_now - self.last_meas_t) > self.stale_timeout_s)

        return GateResult(x, y, meas_changed, data_valid, data_stale)


class AltitudeLock:
    """Hedef Z'sini ani sıçramaları reddeden hareketli ortalamayla kilitler.

    Yalnızca YENİ ölçümler (meas_changed) ortalamaya girer. Kilitli
    ortalamadan jump_reject_cm'den fazla sapan Z sıçraması yok sayılır; ret
    reject_persist kez üst üste sürerse irtifa gerçekten değişmiştir →
    ortalama tazelenir.
    """

    def __init__(self, buffer_size=30, jump_reject_cm=1500.0, reject_persist=8):
        self.buffer_size = buffer_size
        self.jump_reject_cm = jump_reject_cm
        self.reject_persist = reject_persist
        self._buffer = []
        self._reject_count = 0
        self.locked_alt = None

    def update(self, tz, meas_changed, data_valid):
        if meas_changed:
            if self.locked_alt is None or abs(tz - self.locked_alt) < self.jump_reject_cm:
                self._buffer.append(tz)
                if len(self._buffer) > self.buffer_size:
                    self._buffer.pop(0)
                self._reject_count = 0
            else:
                self._reject_count += 1
                if self._reject_count >= self.reject_persist:
                    self._buffer = [tz]
                    self._reject_count = 0

        if self._buffer:
            self.locked_alt = sum(self._buffer) / len(self._buffer)
        elif self.locked_alt is None and data_valid:
            # İlk geçerli ölçüme kadar kilitleme YOK; geçersiz (0) Z ile
            # başlatmak throttle'ı -1'e kilitliyordu.
            self.locked_alt = tz

        return self.locked_alt
