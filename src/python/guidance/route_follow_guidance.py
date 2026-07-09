"""
ELIPS ROTA TAKIP GUDUMU — Carrot Pure-Pursuit + Ayrisik Hiz Kanali
==================================================================
Hedef, ogrenilen elips rota uzerinde izlenir. Dumen (carrot) ile hiz
(yuvaya yetisme) AYRISTIRILIR: hedefin arkasindaki sabit yuva noktasini
DOGRUDAN kovalamak, noktaya varinca kerteriz cokusu -> 360 derece tur
uretiyordu.

  * Halka uzerinde (cross-track <= band): pure pursuit — carrot daima
    dronun kendi fazinin ILERISINDE, kerteriz asla cokmez. Yetisme HIZ
    kanalindan: yuvaya isaretli faz farki -> yay mesafesi -> nominal
    ustu/alti hiz (LPF + yetki siniri ile).
  * Halkaya yaklasma (cross-track > band): yuva noktasina git; sapma
    buyudukce halkanin en yakin noktasina dogru surekli blend.

Hedefin HIZ kanali GUVENILMEZ (bkz. AI_PROJE_OZETI "hiz kanalina guvenme"):
halka taban hizi rotanin kendisinden cikar (|omega|*r_eff = hedefin olculmus
ortalama yol hizi). Kazanclar config/pid_params.yaml -> route_intercept + pursuit.
"""

import math


class GuidePointSlewLimiter:
    """Guide (nişan) noktasının kayma hızını sınırlar: rota yeniden-fit /
    nearest_phi sıçramalarını bir hız tavanıyla yumuşatır, böylece yaw/pitch
    komutlarına basamak girişi olmaz. max_speed_cms genelde nominal hedef
    hızının ~2 katıdır."""

    def __init__(self, max_speed_cms):
        self.max_speed_cms = max_speed_cms
        self._last = None

    def update(self, x, y, dt):
        limit = self.max_speed_cms * dt
        if self._last is not None:
            dx = x - self._last[0]
            dy = y - self._last[1]
            length = math.hypot(dx, dy)
            if length > limit:
                x = self._last[0] + dx * limit / length
                y = self._last[1] + dy * limit / length
        self._last = (x, y)
        return x, y


class RouteFollowGuidance:
    def __init__(self, config):
        ri = config.get("route_intercept", {})
        self.nominal_target_speed = float(ri.get("nominal_target_speed_cms", 1700.0))
        self.follow_dist_cm = config["pursuit"]["follow_distance_cm"]
        self.chase_max = config["pursuit"]["max_speed_cms"]
        self.min_speed = config["pursuit"]["min_speed_cms"]
        self.catch_gain = float(ri.get("catch_gain", 0.6))
        self.ring_v_auth = float(ri.get("ring_v_auth_cms", 1100.0))
        self.ri_follow_cm = float(ri.get("follow_distance_cm", 800.0))
        self.lead_time_s = float(ri.get("lead_time_s", 1.0))
        self.lookahead_phi = float(ri.get("lookahead_phi", 0.25))
        self.cross_track_band = float(ri.get("cross_track_band_cm", 2000.0))

        self.arc_filt = None    # yuvaya yay farkı LPF durumu (hız kanalı)

    def guide(self, route, dx, dy, tx, ty):
        """Aktif (fit edilmiş) rota + dron konumu + gated hedef konumundan
        güdüm noktası üretir.

        Dönüş: (guide_x, guide_y, ring_v_des)
            ring_v_des None ise dron halkaya YAKLAŞMA fazındadır (ileri hız
            guide mesafesinden çizilir); değilse halka üstü faz-farkı hız talebi.
        """
        dir_sign = 1.0 if route.omega >= 0 else -1.0
        r_eff = 0.5 * (route.a + route.b)

        # Takip yuvası cm → faz. Eski sabit 0.35 rad, r_eff ile çarpınca ~35 m
        # geride kalmak demekti ("uzak kalma"nın nedeni).
        trail_phi = min(0.6, max(0.03, self.ri_follow_cm / max(r_eff, 1.0)))

        # GPS DELAY ölçümü ~1 s geriden gösterir (20 m/s hedefte ~20 m!) —
        # hedef fazı rota açısal hızıyla öne alınarak telafi edilir.
        target_phi = route.nearest_phi(tx, ty) + route.omega * self.lead_time_s
        slot_phi = target_phi - dir_sign * trail_phi        # takip yuvası

        drone_phi = route.nearest_phi(dx, dy)
        near_x, near_y = route.point_on_ellipse(drone_phi)
        cross_track_err = math.hypot(dx - near_x, dy - near_y)

        if cross_track_err <= self.cross_track_band:
            # HALKA ÜZERİNDE — pure pursuit carrot, kerteriz asla çökmez
            carrot_phi = drone_phi + dir_sign * self.lookahead_phi
            gx, gy = route.point_on_ellipse(carrot_phi)
            guide_x, guide_y = float(gx), float(gy)

            # Yetişme = hız kanalı: yuvaya işaretli faz farkı → yay → hız
            gap = slot_phi - drone_phi
            while gap > math.pi:
                gap -= 2 * math.pi
            while gap < -math.pi:
                gap += 2 * math.pi
            ahead_arc = dir_sign * gap * r_eff
            # Yuva fazı 1 Hz GPS gürültüsüyle titrer → ~1 s LPF (yalnızca hız
            # kanalı; dümen carrot'u dron konumuna bağlı, zaten temiz).
            if self.arc_filt is None:
                self.arc_filt = ahead_arc
            self.arc_filt += 0.02 * (ahead_arc - self.arc_filt)

            v_route = abs(route.omega) * r_eff
            if not (500.0 <= v_route <= self.chase_max):
                v_route = self.nominal_target_speed
            catch = max(-self.ring_v_auth,
                        min(self.ring_v_auth, self.catch_gain * self.arc_filt))
            ring_v_des = v_route + catch
        else:
            # HALKAYA YAKLAŞMA: yuva noktasına git; sapma büyüdükçe en yakın
            # halka noktasına doğru blend
            gx, gy = route.point_on_ellipse(slot_phi)
            guide_x, guide_y = float(gx), float(gy)
            excess = cross_track_err - self.cross_track_band
            blend = min(1.0, excess / (self.cross_track_band * 2.0))
            guide_x = guide_x * (1.0 - blend) + near_x * blend
            guide_y = guide_y * (1.0 - blend) + near_y * blend
            ring_v_des = None

        return guide_x, guide_y, ring_v_des

    def desired_speed(self, ring_v_des, dist_2d, waiting, data_stale):
        """İleri hız profili. Halka üstünde ring_v_des kullanılır; yaklaşma
        fazında guide mesafesinden çizilir (geride kaldıkça CHASE_MAX'e hızlan)."""
        if ring_v_des is not None:
            v_des = ring_v_des
        else:
            v_des = self.nominal_target_speed + self.catch_gain * (dist_2d - self.follow_dist_cm)
        v_des = max(self.min_speed, min(self.chase_max, v_des))
        if waiting:
            v_des = 0.0             # kilit yok: yerinde bekle
        elif data_stale:
            v_des = self.nominal_target_speed   # yeni ölçüm yok: nominalle düz git
        return v_des
