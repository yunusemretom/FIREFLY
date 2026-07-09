<!--
================================================================================
  Firefly — Route Kamikaze Sistemi: A'dan Z'ye Detaylı Rehber
  Kardeş dosyalar: AI_PROJE_OZETI.md, AI_DOSYA_YAPISI.md
  İlgili kod: tests/python/test_route_kamikaze.py
              src/python/navigation/route_intercept_system.py
              src/python/guidance/pid_{yaw,pitch,throttle,core}.py
              config/pid_params.yaml  (route_kamikaze bölümü)
================================================================================
-->

# 🎯 Route Kamikaze Sistemi — A'dan Z'ye

Bu rehber, `tests/python/test_route_kamikaze.py` dosyasındaki **otonom taarruz
(kamikaze) güdüm sistemini** en ince ayrıntısına kadar açıklar. Amaç: sabit bir
elips/loiter rotasında dönen hedef İHA'yı öğrenip, rota üzerinde en erken
**çarpışma noktasını** çözerek tam hızla o noktaya dalıp hedefi **VURMAK**.

> **Takipten farkı:** `test_route_intercept.py` hedefi *arkasından takip eder*
> (follow_distance ile bir yuvada kalır). Kamikaze **takip etmez** — hedefin
> gelecekteki konumunu kesişim noktasında yakalayıp içinden geçer.

---

## 0. Terimler Sözlüğü (önce bunları oturt)

| Terim | Anlamı |
|---|---|
| **Rota (route)** | Hedefin tekrar tekrar izlediği kapalı elips yörünge |
| **Faz (φ, phi)** | Hedefin elips üzerindeki açısal konumu (rad). `φ=0` büyük eksende. |
| **omega (ω)** | Açısal hız (rad/s). İşareti dönüş yönünü verir (CW/CCW). |
| **Intercept / çarpışma noktası** | Dron düz uçarsa hedefle **aynı anda** buluşacağı elips noktası |
| **Guide noktası** | Dronun o an burnunu çevirdiği hedef nokta (= intercept noktası) |
| **Kerteriz (bearing)** | Dronun guide noktasına bakması gereken yön açısı |
| **Terminal faz** | Son yaklaşma — hedefe belirli bir menzilden yakın |
| **Commit** | "Kararlıyım" fazı — yaw kısılır, düz dalışa geçilir (spin önlenir) |
| **Stale hold** | SDK'nın yeni GPS gelmese bile son değeri tutması |
| **Lock (kilit)** | Rota şeklinin bir tam turdan öğrenilip dondurulması |

---

## 1. Büyük Resim — Sistem Nasıl Çalışır?

```
   ┌────────────────────────────────────────────────────────────────┐
   │  Unreal Simülatör (DronesOfWar.exe)  —  TCP :12345              │
   │  • Dron fiziği  • Hedef İHA (elips loiter)  • Bozuk hedef GPS   │
   └───────────────▲───────────────────────────────┬────────────────┘
                   │ kontrol (throttle,pitch,roll,yaw,arm)          │ telemetri (18+ float / satır)
                   │                                                ▼
   ┌───────────────┴────────────────────────────────────────────────┐
   │  test_route_kamikaze.py  (LiveKamikazeRunner)                   │
   │                                                                 │
   │  ADIM 1: Telemetri al  →  Hedef GPS'i "değişim tespiti"yle süz  │
   │  ADIM 2: Rota öğren     →  RoutePlanner + OrbitLock (elips fit)  │
   │  ADIM 3: Faz izle       →  PLL: gürültüyü süz, hedefi canlı takip│
   │  ADIM 4: Çarpışma çöz    →  phase_intercept() → guide noktası    │
   │  ADIM 5: Güdüm           →  Yaw + Pitch + Throttle PID'leri      │
   │  ADIM 6: Komut gönder   →  set_control_surfaces(...)            │
   │  ADIM 7: Vuruş tespit    →  bağlantı kopması = çarpışma onayı    │
   └─────────────────────────────────────────────────────────────────┘
```

İki çalışma modu vardır:

- **`--sim`** → `KamikazeSimulator`: Simülatör GEREKMEZ. Hedefi ve dronu saf
  Python'da (kinematik nokta) simüle eder. Algoritmayı hızlı doğrulamak için.
- **(varsayılan)** → `LiveKamikazeRunner`: **Gerçek Unreal simülatörüne** TCP ile
  bağlanır, gerçek dron fiziğini uçurur. `--headless` ile GUI'siz koşup CSV +
  rapor PNG üretir.

Çalıştırma:
```bash
python tests/python/test_route_kamikaze.py --sim                       # saf python sim
python tests/python/test_route_kamikaze.py --headless --duration 150   # canlı, GUI'siz
python tests/python/test_route_kamikaze.py                             # canlı + canlı grafik
```

---

## 2. SDK Katmanı — Telemetri ve Kontrol (`drone_sdk.py`)

### 2.1 Bağlantı ve arka plan alım döngüsü
`connect()` bir TCP soketi açar ve **ayrı bir thread**'de `_receive_loop()`
çalıştırır. Bu döngü sürekli satır satır telemetri okuyup `self.telemetry`
sözlüğünü günceller. Ana kontrol döngüsü ise bu sözlükten `get_telemetry()` ile
en güncel değeri okur (kilitli/thread-safe).

### 2.2 Telemetri formatı (bir satır = ≥18 float, virgülle)
```
index  0-2   Dron  position (X, Y, Z)          cm
index  3-5   Dron  rotation (Roll, Pitch, Yaw) derece
index  6-8   Dron  velocity (Vx, Vy, Vz)       cm/s
index  9     Dron  speed (skaler)              cm/s
index 10     Dron  altitude                    cm
index 11-13  Hedef position (X, Y, Z)          cm
index 14-16  Hedef rotation                    derece
index 17     Hedef speed                       cm/s  ← GÜVENİLMEZ, kullanma!
index 18-26  (DEBUG) gerçek/bozulmamış değerler → get_debug_truth()
index 27+    (DEBUG) bozma maskesi + parametreleri
```

### 2.3 İki kritik tuzak
1. **Stale hold:** Hedef GPS max **1 Hz** gelir. Yeni paket gelmese de
   `telemetry["target"]["position"]` **son değeri tutar**. Yani "aynı sayıyı
   50 kez okumak" 50 yeni ölçüm DEĞİLDİR. → Çözüm: **değişim tespiti**
   (`meas_changed = tx != last_tx or ty != last_ty`). Filtreye/rota planlayıcıya
   yalnızca DEĞİŞEN ölçüm beslenir.
2. **Bağlantı kopması ≠ dropout:** Geçici GPS kesintisi (`socket.timeout`)
   bağlantıyı KOPARMAZ (`_receive_loop` `continue` eder). Ancak karşı taraf
   soketi **kapatırsa** (`recv` boş döner) `is_connected=False` olur. Kamikaze'de
   bu, **dron hedefe çarpıp imha oldu** demektir → vuruş onayı (bkz. §9).

### 2.4 Kontrol gönderimi
```python
drone.set_control_surfaces(throttle, pitch, roll, yaw, arm)
```
Değerler **[-1, +1]** aralığında. Önemli: **throttle klasik gaz değil, dikey hız
komutudur** → `-1` = serbest düşüş, `0` = hover, `+1` = max tırmanış.

---

## 3. Rota Modeli — Elips Matematiği (`EllipseRouteModel`)

Hedef tekrar eden bir yörünge izlediği için, her ölçümü tek tek tahmin etmek
yerine **rotanın geometrik şeklini** öğreniriz. Bu, gürültülü 1 Hz veriyi çok
daha sağlam bir tahmine çevirir.

### 3.1 Elips fit (`_fit_ellipse_algebraic`) — Fitzgibbon yöntemi
Girdi: biriken `(x, y)` konum noktaları. Çıktı: elipsin 5 parametresi
`(cx, cy, a, b, θ)` — merkez, yarı-eksenler, dönüklük açısı.

Adımlar:
1. Noktaları ortalamayla merkezle (sayısal kararlılık).
2. Genel konik denklemi `Ax² + Bxy + Cy² + Dx + Ey + F = 0` için
   **elips kısıtlı** (4AC − B² > 0) en-küçük-kareler çöz (özdeğer problemi).
3. Konik katsayılarından geometrik parametreleri çıkar: merkez `(cx,cy)`,
   dönüklük `θ = ½·atan2(B, A−C)`, yarı-eksen uzunlukları `a, b`.
4. **Kanoniklestirme:** her zaman `a ≥ b` (büyük eksen `a`). Aksi halde ardışık
   fit'ler arasında `a↔b` yer değiştirip `θ` 90° zıplar → faz süreksizliği →
   omega/periyot patlar. *(Bu, kod tabanında önceden yaşanmış bir buga karşı
   konmuş bir emniyet.)*

### 3.2 Faz ve açısal hız (omega) kestirimi (`fit`)
Elips bulununca her noktanın **fazı** hesaplanır:
```
xt =  (x−cx)·cosθ + (y−cy)·sinθ      # elips çerçevesine döndür
yt = −(x−cx)·sinθ + (y−cy)·cosθ
φ  = atan2(yt/b, xt/a)               # parametrik faz açısı
```
Fazlar `np.unwrap` ile sürekli hale getirilir, sonra `φ(t) = φ₀ + ω·t` doğrusuna
**lineer regresyon** uygulanır. Eğim = **omega (ω)**, kesişim = **φ₀**.
`ω`'nın işareti dönüş yönünü verir: `ω<0` → CW, `ω>0` → CCW. Periyot `= 2π/|ω|`.

### 3.3 Rota üzerinde konum ve en yakın faz
- `point_on_ellipse(φ)` → verilen faz için `(x, y)` (parametrik denklem +
  θ döndürme).
- `position_at(t)` → model saatine göre `φ = φ₀ + ω(t−t₀)` konumu.
- `nearest_phi(x, y)` → verilen ölçümün elips üzerine **izdüşümü** (360 noktalı
  kaba arama + en yakını). Gürültülü ölçümü "temiz" faza çevirir.

### 3.4 Sağlam fit (`fit_robust`)
1. Tüm veriyle kaba fit. 2. Elipse geometrik uzaklığı `outlier_threshold_cm`
(5000 cm) aşan noktaları at. 3. Temiz veriyle yeniden fit. JUMP/spike ölçümlere
karşı korur.

---

## 4. Rota Planlayıcı ve Yörünge Kilidi

### 4.1 `RoutePlanner` (temel)
- `add_observation(t, x, y)` her **yeni** ölçümde çağrılır (stale değil!).
- Dairesel buffer (varsayılan 60 nokta) tutar; `min_fit_points` (15) birikince
  ilk fit yapılır.
- Fit sonrası her ölçümün **residual**'ini (rota modelinden sapma) izler;
  `residual_threshold`'u (1500 cm) üst üste aşarsa rotayı otomatik yeniden fit
  eder (hedef rota değiştirdiyse).

### 4.2 Yörünge kilidi (kamikaze runner içinde, elle işletilen durum makinesi)
Kamikaze runner sadece `RoutePlanner`'a güvenmez; kendi **360° tur tespiti** ve
**kilitle/dondur** mantığını işletir çünkü intercept çözümü için **kararlı,
donmuş** bir rotaya ihtiyaç var:

1. **İşaretli faz birikimi:** her ölçümde `current_phi = nearest_phi(tx,ty)`.
   Ardışık farklar (±π sarma düzeltilmiş) `phase_acc`'e eklenir.
2. **Tur tamamlandı mı?** `|phase_acc| ≥ 2π` ve elde ≥15 nokta varsa → yeni bir
   `fit_robust` adayı üret.
3. **PERİYOT-KAPSAMA guard'ı** (kritik!): Gürültülü `nearest_phi`, kötü ilk
   fit'le `phase_acc`'i **sahte hızlı** biriktirip <1 turluk çöp veriyle fit'i
   tetikleyebilir (gerçek 29.6 s periyot 11.8 s'e kilitlenip → yanlış omega →
   dron hayalet noktaya koşup spinliyordu). Guard: veri en az `0.75·periyot`
   kapsamıyorsa fit **reddedilir**, veri biriktirilmeye devam edilir.
4. **Tutarlılık kontrolü:** kilitli rota varken yeni adayın omega işareti aynı
   olmalı ve merkezi çok zıplamamalı; değilse reddet.
5. **Dondurma:** `lock_stable_laps` (config, 4) kadar tur üst üste kilitlenince
   `route_frozen=True` → rota artık yeniden fit edilmez (kararlı intercept).

**Neden dondurma?** Donmuş rotanın omega'sındaki minik hata uzun koşuda faz
kaymasına yol açar; bunu **model saati yerine anlık ölçüme demirlenerek** (§5)
telafi ederiz. Yani şekil dondurulur ama hedefin o an NEREDE olduğu her döngü
canlı ölçümden güncellenir.

---

## 5. Faz İzleyici (PLL) — Hedefi Canlı Nişanla

Donmuş elips "şekli" verir ama hedefin o anki **fazını** her döngü bilmek gerek.
Ham `nearest_phi(tx,ty)` GPS gürültüsüyle titrer. Çözüm: bir **PLL (Phase-Locked
Loop)**:

```
phi_meas       = nearest_phi(tx, ty)                # ölçülen faz (gürültülü)
phi_meas_lead  = phi_meas + omega * LEAD_TIME_S     # GPS gecikmesini telafi et
# PLL güncellemesi:
phi_track += omega * dt                             # 1) modelle ilerlet
err        = wrap(phi_meas_lead - phi_track)        # 2) ölçüme kıyasla
phi_track += 0.08 * err                             # 3) küçük katsayıyla düzelt
```

- **`omega*dt` ilerletme:** hedefin nerede olacağını fizikten bilir (yumuşak).
- **`0.08*err` düzeltme:** ölçüme yavaşça çeker → GPS gürültüsünü/OFFSET
  jitterını **alçak geçiren süzgeç** gibi süzer, ama gerçek konumu izler.
- **`LEAD_TIME_S`:** 1 Hz GPS **gecikmelidir**; hedefi omega·lead kadar öne alıp
  "şu an gerçekte nerede" tahminine varırız.

Sonuç: `phi_now = phi_track` → `tgt_now = point_on_ellipse(phi_now)` = hedefin
**tahmini anlık konumu**. `range_to_target = |tgt_now − dron|`.

---

## 6. Çarpışma Noktası Çözücü (`phase_intercept`) ⭐ Sistemin Kalbi

**Problem:** Dron sabit `speed` ile düz uçarsa, hedefle **aynı anda** hangi
noktada buluşur?

**Neden `solve_intercept` (model saatli) DEĞİL?** O, `position_at`
(`φ₀+ω·t`) kullanır; donmuş rotanın omega hatası 30-60 s'de faz kaymasıyla
nişanı gerçek hedeften ~10 m saptırır → dron hayalet noktanın etrafında döner.
`phase_intercept` ise hedefin **anlık ölçüme demirli fazından** (`phi_now`)
başlar.

**Algoritma:**
```
s      = 0, ds, 2ds, ... horizon    # dronun uçacağı süre örnekleri (s)
φ(s)   = phi_now + omega * s         # hedefin s saniye sonraki fazı
hedef(s) = point_on_ellipse(φ(s))   # hedefin s sonra olacağı yer
dist(s)  = |hedef(s) − dron|          # o ana hedefe uzaklık
f(s)   = speed * s − dist(s)          # dron s'de bu kadar yol alır; farkı sıfırla
```
`f(s) = 0` kökü = dronun hedefe tam yetişebildiği ilk an `s*`.
`f`'nin **artı yönde** işaret değiştirdiği ilk aralık bulunur (dron başta
geride, sonra yetişir), lineer interpolasyonla `s*` inceltilir. Çarpışma noktası
`= point_on_ellipse(phi_now + omega·s*)`.

**Kritik gözlem:** `f(s) = speed·s − dist(s)` kökünün var olması için dronun
**kapanma hızı pozitif** olmalı. Dron hızı ≲ hedef hızıysa `f` hiç sıfırı
geçmez → `None` döner → yakalama imkânsız. **Bu, "hedefi hiç vuramıyor" sorununun
kök nedenidir** (bkz. §11).

Kök yoksa (`None`): hedefin anlık tahmini konumuna (`tgt_now`) **saf takip**
(ram) yapılır — en azından peşinden gider.

---

## 7. Güdüm Zinciri — Nokta'dan Stick Komutuna

Guide noktası `(guide_x, guide_y)` + kilitli irtifa belliyken, üç bağımsız
kontrolcü stick komutlarını üretir.

### 7.0 Guide eğim-sınırı (slew limit)
Intercept/nearest_phi çözümündeki ani sıçramalar guide noktasını zıplatabilir.
Guide, taarruz hızının ~2 katıyla (`KAMIKAZE_SPEED*2*dt`) sınırlanıp yumuşatılır
→ kontrolcülere pürüzsüz hedef girer.

### 7.1 YAW — Kerteriz kontrolü (`YawController`, `pid_yaw.py`)
```
target_bearing = atan2(guide_y − dy, guide_x − dx)   # gitmesi gereken yön
yaw_error      = wrap_180(target_bearing − drone_yaw)
yaw_cmd        = PID(yaw_error)                        # [-1, +1]
```
Basit ama kritik: dronun burnunu guide noktasına çevirir. `inner_pid.yaw`
kazançları: `kp=0.035, ki=0.0005, kd=0.002`.

### 7.2 PITCH — Hız + irtifa açısı (`PitchController`, `pid_pitch.py`)
İki katmanlı:
1. **Dış hız PID'i:** `speed_error = v_des − v_fwd` (m/s). Burada `v_fwd`
   **imzalı ileri hız** — hız vektörünün burun yönüne izdüşümü
   (`Vx·cos(yaw)+Vy·sin(yaw)`). Skaler `|speed|` geriye kaçışı gizlerdi, bu yüzden
   izdüşüm kullanılır. Çıktı `speed_adj` (istenen eğim düzeltmesi).
2. **İstenen eğim:** `raw = target_elevation − speed_adj`. `target_elevation =
   atan2(ldz, dist_2d)` (guide'a bakış açısı). Slew limit + [-1,1] normalize.
3. **Yaw hizalama ölçeği:** `pitch *= max(0, cos(yaw_error))`. Dron hedefe henüz
   dönmediyse (yaw_error büyük) ileri gaz kısılır → yana kayarak (crabbing)
   yol almaz, önce döner.

### 7.3 THROTTLE — Eş-irtifa kilidi (`ThrottleController`, `pid_throttle.py`)
```
dz_m  = clamp((locked_alt − drone_z)/100, −5, 5)   # irtifa hatası (m)
thr   = base_throttle(0) + alt_PID(dz_m)            # dikey hız komutu
```
`locked_alt` = hedefin **sıçrama reddeden hareketli ortalama irtifası**
(§8). Bu, 2B kesişimi **3B çarpışmaya** çevirir. Kamikaze runner throttle
yetkisini yükseltir (`kp=0.40, ki=0.06, int_limit=12`) çünkü hızlı ileri uçuşta
burun eğimi tırmanma coupling'i üretiyor ve varsayılan yetki kalıcı ~2 m irtifa
hatası bırakıyordu.

### 7.4 PID çekirdeği (`pid_core.py`)
Standart PID: `out = kp·e + ki·∫e + kd·de/dt`, integral wind-up limiti ve çıktı
clamp'i ile. `SlewLimiter` değişim hızını sınırlar. `wrap_180` açı sarma.

### 7.5 İşaret düzeni ve gönderim
```python
fy    =  clamp(yaw_cmd, -1, 1)
fp    = -clamp(pitch_cmd, -1, 1)   # sim_bridge işaret düzeni: pitch ters
f_thr =  clamp(thr_cmd, -1, 1)
drone.set_control_surfaces(f_thr, fp, 0.0, fy, True)   # roll=0, arm=True
```

---

## 8. İrtifa Kilidi ve Terminal Dalış

### 8.1 Kilitli irtifa (`locked_alt`)
Hedefin gerçek irtifası pek değişmediği için Z, **sıçrama reddeden hareketli
ortalamayla** tutulur (buffer 30, `ALT_JUMP_REJECT_CM=1500`). Ani bir Z sıçraması
8 kez üst üste ısrar etmedikçe reddedilir (JUMP'a karşı). Böylece 2B kesişim
sabit bir yükseklikte 3B çarpışmaya oturur.

### 8.2 Terminal dalış bias'ı (`TERMINAL_DESCENT_CM`)
Son yaklaşmada `ldz` (irtifa hatası) kademeli olarak azaltılıp hedefin biraz
ALTINA nişan alınabilir (`ldz −= TERMINAL_DESCENT_CM · frac`), böylece dron
irtifadan geçerek çarpar. **DİKKAT:** Bu bias yalnızca dron hedefin ÜSTÜNDE
kalma eğilimindeyse doğrudur. Ölçümlerde dron zaten hedefin **altında** kalınca
bias ıskalayı büyütür → güncel config'te **0'a** çekildi (bkz. §11).

---

## 9. Vuruş Tespiti — "Bağlantı Kopması = Çarpışma"

Kamikaze'de mesafe yarıçapıyla erken bitirme **YOKTUR**. Mantık:

1. **`attack_started`**: rota kilitlenip bekleme bitince taarruz fiilen başlar.
2. Simülatör, dron hedefe **fiziksel çarpınca** soketi kapatır →
   `drone.is_connected()` False olur.
3. Bu kopma:
   - `attack_started` **ve** dron hedefe yakınken (`last_range < CONFIRM_RANGE`,
     2000 cm) → **🎯 ÇARPIŞMA ONAYLANDI**.
   - `attack_started` ama dron uzaktayken → "sim donması", çarpışma sayılmaz.
   - Taarruz başlamadan koptuysa → koşu iptal.

**Rapor metrikleri** (headless, `get_debug_truth()` ile): en yakın gerçek geçiş
mesafesi, tahmini geçiş mesafesi, toplam gövde dönüşü (spin göstergesi), yaw hızı
RMS, irtifa hatası RMS, terminal faz oranı. Bunlar CSV + PNG'ye yazılır.

---

## 10. Ana Kontrol Döngüsü — Adım Adım (LiveKamikazeRunner.run)

Her `DT_LOOP` (0.02 s = 50 Hz) döngüde:

```
1.  Bağlantı koptu mu?  → evetse vuruş/iptal kararı ver, çık.
2.  Telemetri al: dron pos/rot/vel, hedef pos.
3.  Değişim tespiti: meas_changed? (stale hold'a karşı) + JUMP reddi.
4.  İrtifa kilidini güncelle (sıçrama reddi).
5.  meas_changed ise: orbit buffer + RoutePlanner'a ölçüm besle.
6.  Rota fitli mi?
      ├─ Faz birikimi güncelle → 360° tur → kilit/dondur denemesi (guard'lı).
      ├─ PLL ile phi_now → tgt_now, range_to_target hesapla.
      ├─ terminal = range < TERMINAL_RANGE ?
      ├─ commit  = range < COMMIT_RANGE  (LATCH'li, §11) ?
      └─ phase_intercept → guide noktası (yoksa ram takip).
7.  Guide slew-limit.
8.  ldx,ldy,ldz hesapla (ldz = locked_alt − dz, terminal bias uygula).
9.  waiting?  (kilit yok → hover, yaw=0, v_des=0)
10. YAW  = YawController(ldx, ldy).  commit'te CAP'le (anti-spin).
11. PITCH = PitchController(v_des, v_fwd, elevation).
12. THR   = ThrottleController(ldz).
13. set_control_surfaces(thr, -pitch, 0, yaw, arm).
14. Vuruş metriklerini/CSV'yi güncelle.
15. Süre doldu mu? GUI güncelle. dt uyumu için uyu.
```

---

## 11. Bilinen Sorunlar ve Yapılan Düzeltmeler ⚠️

Bu bölüm, "hedefi ya çok geç ya hiç vuramıyor" şikâyetinin kök nedenlerini ve
uygulanan düzeltmeleri kaydeder.

### 🔴 Sorun 1 — Taarruz hızı hedeften düşük/eşit → intercept ÇÖZÜLEMİYOR
**Kök neden:** `route_kamikaze.attack_speed_cms = 1800` idi; hedef hızı da
~1500-1800 cm/s. `phase_intercept`'in `f(s)=speed·s−dist(s)` kökü ancak
**kapanma hızı pozitif** olunca vardır. Hız farkı ~0 olunca çözücü sürekli
`None` döndü → dron intercept yerine sadece hedefin peşinden **teğet geçişler**
yaptı, hiç kesişemedi. Ölçümde terminal faz oranı yalnızca **%5** çıktı (dron
neredeyse hiç yakına gelemiyor).

**Düzeltme:** `attack_speed_cms: 1800 → 2800` (config). Kapanma hızı ~1000 cm/s
oldu, intercept çözülebilir hale geldi. Terminal faz oranı **%5 → ~%50**; dron
artık her turda hedef bölgesine tekrar tekrar ulaşıyor.

### 🔴 Sorun 2 — Geçiş sonrası SPIN (dron kendi etrafında dönüyor)
**Kök neden:** Dron intercept noktasını geçince menzil tekrar büyüyüp COMMIT
menzilinden çıkıyordu; yaw yetkisi anında açılıyor, kerteriz 180° savruluyor ve
dron **tam spin** atıyordu (150 s'de ~16-18 gövde dönüşü, yaw RMS ~82 deg/s).

**Düzeltme — Commit LATCH:** `commit_latched` bayrağı eklendi. Dron bir kez
COMMIT menziline (`< COMMIT_RANGE`) girince **kilitlenir**; yalnızca menzil
terminal ufkun (`> TERMINAL_RANGE`) üstüne net çıkınca serbest kalır. Latch'liyken
yaw kısık kalır → dron geçişten sonra spinlemek yerine **düz dalışta kalıp**
geniş bir kavisle tekrar yaklaşır. Sonuç: gövde dönüşü ~18 → ~11-13, yaw RMS
~82 → ~60 deg/s.

### 🔴 Sorun 3 — Terminal dalış bias'ı yanlış yönde
**Kök neden:** `TERMINAL_DESCENT_CM=400` dronu daima hedef irtifasının **ALTINA**
nişanlatıyordu; oysa ölçümde dron zaten zamanın ~%70'inde hedefin altında
kalıyordu (burun-eğimi/throttle coupling'i yüzünden). Bias, dikey ıskalayı
büyütüyordu.

**Düzeltme:** `terminal_descent_cm: 400 → 0` (config, açıkça eklendi). Eş-irtifa
nişanı çarpışmayı dikeyde ortalıyor; irtifa hatası RMS düştü.

### 🟡 Not — Değerlendirmede koşu-arası varyans
Canlı Unreal simülatörü **deterministik değildir**; her koşuda hedef fazı, fit
zamanlaması ve fizik biraz farklı olur. Tek bir koşunun min-mesafesi gürültülüdür
(ör. aynı config 155 cm ↔ 241 cm arası oynayabilir). Bir değişikliğin gerçekten
iyileştirdiğini anlamak için **birkaç koşunun** min-mesafe/dönüş dağılımına
bakılmalı, tek koşuya değil. (Başlangıçtaki tek-kare dron telemetri spike'ları
irtifa RMS'ini şişirebilir — bunlar kontrol ıraksaması değildir.)

---

## 12. Konfigürasyon Referansı (`config/pid_params.yaml → route_kamikaze`)

| Parametre | Değer | Anlamı |
|---|---|---|
| `attack_speed_cms` | 2800 | Taarruz (kapanma) hızı. **Hedef hızının belirgin üstünde olmalı** yoksa intercept çözülemez. |
| `terminal_range_cm` | 3000 | Bu menzil altında terminal dalış kademeli başlar. |
| `hit_radius_cm` | 500 | Rapor metriği: en yakın geçiş bununla kıyaslanır. |
| `terminal_descent_cm` | 0 | Terminal irtifa bias'ı. 0 = eş-irtifa (hedefin gerçek yüksekliğine dal). |
| `lead_time_s` | 1.0 | GPS gecikme telafisi (faz omega·lead öne alınır). |
| `wait_for_lock` | true | İlk yörünge kilidine kadar hover'da bekle. |
| `lock_stable_laps` | 4 | Bu kadar tur kilit sonrası rota dondurulur. |

Kod içi (config'te olmayan, `rk_cfg.get` varsayılanları):
`CONFIRM_RANGE_CM=2000`, `COMMIT_RANGE_CM=1500`, `YAW_COMMIT_CAP=0.20`,
`STALE_TIMEOUT_S=2.5`, `OUTLIER_THRESHOLD_CM=5000`, `ALT_BUFFER_SIZE=30`.

---

## 13. Hızlı Sorun Giderme

| Belirti | Olası neden | Bakılacak yer |
|---|---|---|
| Dron hedefe hiç yaklaşmıyor | attack_speed ≤ hedef hızı → intercept `None` | §6, §11-Sorun1 |
| Dron hedefin etrafında dönüyor (spin) | Commit latch yok / gevşek; ya da yanlış omega | §11-Sorun2, §4.2 guard |
| Dron sürekli hedefin altından/üstünden geçiyor | Terminal bias yanlış yön; throttle yetkisi az | §8, §7.3 |
| "Yörünge kilitlenmiyor" | Periyot-kapsama guard reddediyor (veri <0.75·periyot) | §4.2 |
| Vuruş onaylanmıyor ama dron çok yaklaşıyor | Sim çarpışma kutusu küçük; min mesafe hâlâ >~1 m | §9, §11-Not |
| İrtifa RMS aşırı büyük | Başlangıç telemetri spike'ı (tek kare) | §11-Not |

---

*Son güncelleme: 2026-07-09 — bu rehber, kamikaze güdüm sistemi analizi ve
attack_speed / commit-latch / terminal-descent düzeltmeleri sırasında yazıldı.*
