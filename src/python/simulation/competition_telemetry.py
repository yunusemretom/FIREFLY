"""
YARISMA MODU TELEMETRI FILTRESI
================================
Simulator, yarismada VERILMEYECEK kanallari da gonderir (hedef rotation, dron
hiz vektoru, ~100 Hz). Bu sinif sim telemetrisinden SADECE sartnamede
tanimlanan alanlari, sartnamedeki hizlarda gecirir. Kontrol yolu telemetriyi
yalnizca buradan okumalidir.

Sartname (Haberlesme paketi ve limit bilgileri):
  Avci dron : GPS konum, yer hizi (skaler), durus (roll, pitch, yaw)
              hiz TX ayarina bagli: 3.125 Hz (1:32) ... 50 Hz (1:2)
  Hedef IHA : enlem, boylam, irtifa (kalkis noktasina gore), yer hizi; <= 5 Hz

Konum sim'de yerel cm (X, Y, Z) olarak gelir; yarismada LLA gelecek ve yerel
ENU'ya cevrilecek (bkz. geo donusumu). Algoritmalar yerel cm ile calisir.
"""

import math


class CompetitionTelemetry:
    def __init__(self, drone_hz=50.0, target_hz=5.0):
        self.drone_period = 1.0 / drone_hz
        self.target_period = 1.0 / target_hz
        self._drone = None
        self._drone_t = -1e9
        self._target = None
        self._target_t = -1e9
        self._last_target_raw = None

    def update(self, t, raw):
        """raw: drone_sdk.get_telemetry() sozlugu. -> filtrelenmis kopya.

        Donus: {"drone": {...,"stamp", "fresh"}, "target": {..., "stamp", "fresh"}}
        `fresh` bu cagrida yeni paket geldiyse True.
        """
        d, g = raw["drone"], raw["target"]
        drone_fresh = False
        if t - self._drone_t >= self.drone_period - 1e-6:
            vx, vy, _ = d["velocity"]
            self._drone = {
                "position": tuple(d["position"]),
                "ground_speed": math.hypot(vx, vy),
                "rotation": tuple(d["rotation"]),
                "stamp": t,
            }
            self._drone_t = t
            drone_fresh = True
        target_fresh = False
        g_pos = tuple(g["position"])
        if g_pos != self._last_target_raw and t - self._target_t >= self.target_period - 0.02:
            self._last_target_raw = g_pos
            self._target = {"position": g_pos, "speed": float(g["speed"]), "stamp": t}
            self._target_t = t
            target_fresh = True
        return {
            "drone": dict(self._drone, fresh=drone_fresh),
            "target": dict(self._target, fresh=target_fresh) if self._target else None,
        }
