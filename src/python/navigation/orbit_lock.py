"""
YORUNGE KILIDI — Tekrarlayan Rotayi Ogrenip Dondurma Durum Makinesi
====================================================================
Hedef tekrarlayan bir elips rota (loiter/racetrack) izler. Bu modul, canli
olcumlerden bir tam turu (360 derece) tespit edip robust elips fit ile
"master" rotayi kilitler; birkac kararli tur sonrasi rotayi DONDURUR (her
turda kor re-fit, elips parametrelerini oynatip nisan/guide noktasini
basamakli zplatiyordu — tur 5-6 titremesinin kok nedeni).

Faz birikimi ISARETLI toplanir: |delta| toplamak GPS gurultu jitterini de
sayip yarim turda sahte "tur tamamlandi" uretiyordu (erken/yanlis kilit).

period_coverage guard'i (opsiyonel): gurultulu nearest_phi + kotu ilk fit,
phase_acc'i sahte hizli biriktirip <1 turluk veriyle fit tetikleyebilir
(gercek 29.6 s periyot 11.8 s kilitlenip yanlis omega -> dron hayalet
noktaya kosup spin). Veri en az period_coverage * periyot kadar zaman
kapsamiyorsa fit reddedilir. > 0 iken reddedilen fit'te buffer KORUNUR
(veri birikip bir tam tura ulasinca fit iyilesebilsin); 0 iken (klasik
davranis) her tetikte buffer bosaltilir.

RoutePlanner'i (route_intercept_system) icerir; active_route kilitliyse
master, degilse planner cktisidir.
"""

import math
import numpy as np

from .route_intercept_system import RoutePlanner, EllipseRouteModel


class OrbitLockTracker:
    def __init__(self, buffer_size=60, min_fit_points=15, residual_threshold=1500.0,
                 outlier_threshold_cm=5000.0, lock_stable_laps=3,
                 min_lap_points=15, period_coverage=0.0, on_lock=None, verbose=True):
        self.planner = RoutePlanner(buffer_size=buffer_size,
                                    min_fit_points=min_fit_points,
                                    residual_threshold=residual_threshold)
        self.outlier_threshold_cm = outlier_threshold_cm
        self.lock_stable_laps = lock_stable_laps
        self.min_lap_points = min_lap_points
        self.period_coverage = period_coverage
        self.on_lock = on_lock          # kilit anında çağrılır (ör. PLL sıfırlama)
        self.verbose = verbose

        self.master_route = EllipseRouteModel()
        self.orbit_locked = False
        self.route_frozen = False
        self.orbit_count = 0

        self._buf_t, self._buf_x, self._buf_y = [], [], []
        self._phase_acc = 0.0
        self._last_phi = None

    @property
    def active_route(self):
        """Kilitliyse dondurulan master, değilse planner'ın canlı fit'i."""
        return self.master_route if self.orbit_locked else self.planner.route

    @property
    def last_residual(self):
        return self.planner.last_residual

    def step(self, t_now, tx, ty, meas_changed):
        """Bir kontrol döngüsünde çağrılır. Yalnızca meas_changed=True iken
        planner ve tur buffer'ı yeni ölçümle beslenir (bayat verinin her
        döngüde tekrar eklenmesi fit'i o noktaya çeker). Aktif rotayı döndürür.
        """
        if meas_changed:
            self._buf_t.append(t_now)
            self._buf_x.append(tx)
            self._buf_y.append(ty)
            self.planner.add_observation(t_now, tx, ty)

        route = self.active_route
        if not route.fitted:
            return route

        # Faz birikimi (360° tespiti) — İŞARETLİ
        phi = route.nearest_phi(tx, ty)
        if self._last_phi is not None:
            delta = phi - self._last_phi
            if delta > math.pi:
                delta -= 2 * math.pi
            elif delta < -math.pi:
                delta += 2 * math.pi
            self._phase_acc += delta
        self._last_phi = phi

        if abs(self._phase_acc) >= 2 * math.pi and len(self._buf_t) >= self.min_lap_points:
            self._try_lock(tx, ty)

        return self.active_route

    def _try_lock(self, tx, ty):
        committed = False
        if not self.route_frozen:
            try:
                candidate = EllipseRouteModel()
                candidate.fit_robust(
                    np.array(self._buf_t),
                    np.array(self._buf_x),
                    np.array(self._buf_y),
                    outlier_threshold_cm=self.outlier_threshold_cm,
                )
                accept = True
                reason = ""

                # PERİYOT-KAPSAMA guard (yalnızca period_coverage>0)
                t_span = self._buf_t[-1] - self._buf_t[0]
                if self.period_coverage > 0.0 and (
                        candidate.period is None
                        or t_span < self.period_coverage * candidate.period):
                    accept = False
                    reason = (f"veri {t_span:.0f}s < {self.period_coverage:.2f}·periyot "
                              f"({candidate.period:.0f}s) — kapsama yetersiz")
                elif self.orbit_locked:
                    # Dejenerelik: yön (omega işareti) ters dönmemeli, merkez
                    # aşırı sıçramamalı; aksi halde eski kilit korunur.
                    center_jump = math.hypot(
                        candidate.cx - self.master_route.cx,
                        candidate.cy - self.master_route.cy)
                    if np.sign(candidate.omega) != np.sign(self.master_route.omega):
                        accept = False
                        reason = "yön (omega işareti) tutarsız"
                    elif center_jump > self.outlier_threshold_cm:
                        accept = False
                        reason = f"merkez {center_jump:.0f}cm zıpladı"

                if accept:
                    self.master_route = candidate
                    self.orbit_locked = True
                    self.orbit_count += 1
                    committed = True
                    if self.orbit_count >= self.lock_stable_laps:
                        self.route_frozen = True
                    if self.on_lock is not None:
                        self.on_lock()
                    if self.verbose:
                        print(f"  ★ Yörünge kilitlendi! (tur #{self.orbit_count}, "
                              f"{len(self._buf_t)} nokta, "
                              f"periyot≈{self.master_route.period:.1f}s"
                              f"{' — DONDU' if self.route_frozen else ''})")
                elif self.verbose:
                    if self.period_coverage > 0.0:
                        print(f"  ⚠ Yeni fit reddedildi ({reason}) — veri biriktirmeye devam.")
                    else:
                        print("  ⚠ Yeni fit reddedildi (yön/merkez tutarsız) "
                              "— önceki kilit korunuyor.")
            except ValueError as e:
                if self.verbose:
                    print(f"  ✗ Robust fit başarısız: {e}")

        # Sonraki tur için faz sıfırla, çerçeveyi yeniden demirle
        self._phase_acc = 0.0
        self._last_phi = self.active_route.nearest_phi(tx, ty)

        # Buffer yönetimi: fit kabul edildiyse/rota donduysa boşalt. Aksi
        # halde period_coverage>0 iken KORU (bir tam tura ulaşsın), =0 iken
        # (klasik intercept davranışı) her tetikte boşalt.
        keep_buf = (self.period_coverage > 0.0
                    and not committed and not self.route_frozen)
        if keep_buf:
            if len(self._buf_t) > 600:   # bellek/performans emniyeti
                drop = len(self._buf_t) // 4
                del self._buf_t[:drop]
                del self._buf_x[:drop]
                del self._buf_y[:drop]
        else:
            self._buf_t.clear()
            self._buf_x.clear()
            self._buf_y.clear()
