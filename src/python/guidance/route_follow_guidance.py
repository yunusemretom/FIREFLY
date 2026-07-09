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
        # catch_damp: yay farkı türevine (kapanma hızına) sönümleme. P-only
        # kanal (LPF + hız PID gecikmesiyle) yuva etrafında sönümsüz salınım
        # üretiyordu: v_des std ~800 cm/s, mesafe p2p ~40 m ölçüldü.
        self.catch_damp = float(ri.get("catch_damp", 1.2))
        self.arc_tau_s = float(ri.get("arc_tau_s", 0.5))
        self.ring_v_auth = float(ri.get("ring_v_auth_cms", 1100.0))
        self.ri_follow_cm = float(ri.get("follow_distance_cm", 800.0))
        self.lead_time_s = float(ri.get("lead_time_s", 1.0))
        self.lookahead_phi = float(ri.get("lookahead_phi", 0.25))
        self.cross_track_band = float(ri.get("cross_track_band_cm", 2000.0))

        self.arc_filt = None    # yuvaya yay farkı LPF durumu (hız kanalı)
        self.arc_rate = 0.0     # filtreli yay farkı türevi (cm/s)

    def guide(self, route, dx, dy, tx, ty, dt=0.02):
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

        # --- HALKA kanalı (daima hesaplanır) ---------------------------
        # pure pursuit carrot: daima dronun fazının ilerisinde, kerteriz çökmez
        carrot_phi = drone_phi + dir_sign * self.lookahead_phi
        cx_, cy_ = route.point_on_ellipse(carrot_phi)

        # Yetişme = hız kanalı: yuvaya işaretli faz farkı → yay → hız
        gap = slot_phi - drone_phi
        while gap > math.pi:
            gap -= 2 * math.pi
        while gap < -math.pi:
            gap += 2 * math.pi
        ahead_arc = dir_sign * gap * r_eff
        # Yuva fazı 1 Hz GPS gürültüsüyle titrer → LPF (yalnızca hız
        # kanalı; dümen carrot'u dron konumuna bağlı, zaten temiz).
        # Türev, LPF ÇIKIŞINDAN alınır (zaten pürüzsüz) → PD sönümleme.
        alpha = 1.0 - math.exp(-dt / max(self.arc_tau_s, 1e-3))
        if self.arc_filt is None:
            self.arc_filt = ahead_arc
        prev_filt = self.arc_filt
        self.arc_filt += alpha * (ahead_arc - self.arc_filt)
        raw_rate = (self.arc_filt - prev_filt) / max(dt, 1e-3)
        self.arc_rate += 0.1 * (raw_rate - self.arc_rate)

        # Halka taban hızı: elipste yol hızı sabit değildir —
        # v(φ)=|ω|·√(a²sin²φ+b²cos²φ), a/b=1.5'te ±%20 oynar. Ortalama
        # (|ω|·r_eff) kullanmak hız kanalına periyodik bozucu sokuyordu;
        # dronun bulunduğu fazdaki YEREL hız kullanılır.
        sin_p, cos_p = math.sin(drone_phi), math.cos(drone_phi)
        v_route = abs(route.omega) * math.hypot(route.a * sin_p, route.b * cos_p)
        if not (500.0 <= v_route <= self.chase_max):
            v_route = self.nominal_target_speed
        catch = self.catch_gain * self.arc_filt + self.catch_damp * self.arc_rate
        catch = max(-self.ring_v_auth, min(self.ring_v_auth, catch))
        v_ring = v_route + catch

        # --- YAKLAŞMA kanalı + SÜREKLİ harmanlama ----------------------
        # Eski keskin if/else (band içi: carrot, band dışı: yuva noktası)
        # band sınırında MOD TİTREŞİMİ üretiyordu: canlı koşuda 132 s'de 26
        # sınır geçişi ölçüldü; her geçişte guide noktası ve hız formülü
        # zıplayıp yaw patlaması (70 deg/s RMS pencereleri) tetikliyordu.
        # w: 0 = tam halka modu, 1 = tam yaklaşma modu; smoothstep ile sürekli.
        w = (cross_track_err - self.cross_track_band) / self.cross_track_band
        w = max(0.0, min(1.0, w))
        w = w * w * (3.0 - 2.0 * w)

        if w <= 0.0:
            guide_x, guide_y = float(cx_), float(cy_)
            ring_v_des = v_ring
        else:
            # yaklaşma noktası: yuva noktası, sapma büyüdükçe en yakın halka
            # noktasına doğru kayar
            sx, sy = route.point_on_ellipse(slot_phi)
            far_blend = min(1.0, (cross_track_err - self.cross_track_band)
                            / (self.cross_track_band * 2.0))
            ax = float(sx) * (1.0 - far_blend) + near_x * far_blend
            ay = float(sy) * (1.0 - far_blend) + near_y * far_blend
            guide_x = float(cx_) * (1.0 - w) + ax * w
            guide_y = float(cy_) * (1.0 - w) + ay * w
            # hız da harmanlanır: yaklaşma hızı guide mesafesinden çizilir
            dist_g = math.hypot(guide_x - dx, guide_y - dy)
            v_appr = (self.nominal_target_speed
                      + self.catch_gain * (dist_g - self.follow_dist_cm))
            ring_v_des = v_ring * (1.0 - w) + v_appr * w

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
