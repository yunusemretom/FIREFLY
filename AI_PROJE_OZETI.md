<!--
================================================================================
  YAPAY ZEKA ARAÇLARI İÇİN NOT (AI ASSISTANT DIRECTIVE)
================================================================================
  Bu dosya Firefly projesinin OTORİTE (authoritative) özetidir. Bir yapay zeka
  aracıysan (Claude, Cursor, Copilot, vb.):
    1. Projeye dair bağlam için ÖNCE bu 3 dosyayı oku, kod tabanını baştan
       tarayarak yeniden analiz ETME:
         - AI_PROJE_OZETI.md   (bu dosya)  → ne yaptığımız, mimari, algoritmalar
         - AI_DOSYA_YAPISI.md              → hangi dosya ne işe yarar
         - AI_GOREVLER.md                  → yapılan / yapılacak görevler
    2. Bu dosyalar güncel değilse veya kodla çeliştiğini fark edersen, ilgili
       satırı doğrula ve GÜNCELLE — böylece bir sonraki araç tekrar taramasın.
  Son güncelleme: 2026-07-04
================================================================================
-->

# 🔥 Firefly — Proje Özeti

**Teknofest "Drone of War" yarışması** için otonom kamikaze/takip drone kontrol
sistemi. Unreal Engine tabanlı simülatöre **TCP** üzerinden bağlanır; gürültülü
ve düşük frekanslı (1 Hz) hedef GPS'inden **sensör füzyonu** (EKF/UKF) ile temiz
bir hedef durumu kestirir ve **güdüm (guidance) + PID** ile hedefi kovalayıp
yakalar.

---

## 1. Amaç ve Senaryo

- **Avcı (hunter) drone** = quadcopter; SDK ile 4 eksen (throttle, pitch, roll,
  yaw) + arm komutu gönderilir.
- **Hedef** = sabit kanatlı **Talon İHA** (kanat açıklığı 1718 mm). Yerinde
  dönemez; kavisli (banked-turn) manevra yapar. Sabit/tekrarlayan bir rota
  (loiter / racetrack) izler ve hızı genelde ~1500 cm/s civarındadır.
- **Zorluk:** Hedefin GPS telemetrisi **en fazla 1 Hz** gelir ve şu bozulmalara
  maruzdur (SDK'daki `FLAG_*` bit maskesiyle temsil edilir):
  `NOISE` (konum gürültüsü), `SPEEDNOISE`, `OFFSET` (sabit kayma),
  `JUMP` (ani spike), `DROPOUT` (veri kesintisi), `RATELIMIT` (max 1 Hz),
  `DELAY` (gecikmeli veri).
- **Kritik gözlem:** SDK telemetri alanı yeni paket gelmese de **son değeri
  tutar** (stale hold). Bu yüzden ham veriyi zamanlayıcıyla filtreye beslemek
  yanlıştır → sadece **değer değiştiğinde** ölçüm işlenmelidir.

---

## 2. Sistem Mimarisi

```
┌──────────────────────────┐   TCP :12345   ┌───────────────────────────┐
│  Python Beyin (bu repo)   │◄──────────────►│  Unreal Engine Simülatör   │
│                            │  kontrol CSV   │  (DronesOfWar.exe)         │
│  drone_sdk.py (TCP client) │───────────────►│  Drone fizik + Hedef sist. │
│      ▲                     │  telemetri CSV │                            │
│      │ telemetri           │◄───────────────│                            │
│  ┌───┴────────────────┐    │                └───────────────────────────┘
│  │ navigation (EKF/UKF)│   │
│  │ guidance (pursuit)  │   │   ┌──────────────────────────────┐
│  │ vision (RF-DETR)    │   │   │  GCS (Node.js/Electron + Web) │
│  └────────────────────┘    │◄─►│  server.js: TCP→WebSocket köprü│
└──────────────────────────┘     │  tarayıcı arayüzü / telemetri  │
                                  └──────────────────────────────┘
```

**Katmanlar:**
1. **SDK katmanı** (`drone_sdk.py`) — TCP soket, telemetri parse, kontrol
   gönderimi. `get_target_location()`, `get_drone_location()`,
   `set_control_surfaces()`, `get_debug_truth()` (test için gerçek değerler).
2. **Navigasyon / füzyon** (`src/python/navigation/`) — gürültülü GPS'ten temiz
   hedef durumu (konum, hız, yön, dönüş hızı) kestirir.
3. **Güdüm** (`src/python/guidance/`) — kestirilen duruma göre nişan/lead noktası
   üretir, PID ile stick komutlarına çevirir, mod geçişlerini yönetir.
4. **Görü** (`src/python/vision/`) — RF-DETR ile hedef tespiti (TensorRT/ONNX
   export ve inference araçları dahil). Şu an iskelet + inference scriptleri.
5. **GCS** (`src/gcs/`) — Yer kontrol istasyonu; hem Electron hem tarayıcı
   tabanlı arayüz, TCP↔WebSocket köprüsü.
6. **Eski masaüstü GUI** (`drone_gui.py`) — Tkinter tabanlı manuel/kamikaze
   kontrol paneli ve veri toplama (Excel loglama). Hâlâ çalışır durumda.

---

## 3. Ana Algoritmalar

### 3.1 CTRV UKF Hedef Takibi — `navigation/fixed_wing_tracker.py`  ⭐ (aktif)
- **Model:** CTRV (Constant Turn Rate & Velocity). 7 durum:
  `[x, y, yaw, v, yaw_rate, z, vz]`. Ölçüm: sadece GPS `[x, y, z]`.
- **Neden UKF:** filterpy `UnscentedKalmanFilter`; kavisli dönüşlerde CV
  Kalman'dan daha düşük hata. 1 Hz'de ham veriye göre **~%27 daha düşük** konum
  hatası (5 rastgele tohumla doğrulandı).
- **Kalibrasyon:** `hedef_konumlari.xlsx` gerçek vs gürültülü verisinden ölçülen
  gürültü std'si `R = (1426.5, 1574.9, 385.1)`.
- **Sağlamlık:** `_sanitize_state()` — hızı `V_BOUNDS = (300, 4000) cm/s`,
  dönüş hızını `YAW_RATE_MAX = 0.6 rad/s` içinde tutar; negatif hız (heading'in
  180° kilitlenmesi) durumunu flip ile düzeltir. Bu, UKF'nin nadir dejenere
  çözüme kaçıp lead-time'ı patlatmasını önler.
- **API:** `init()`, `update(t,x,y,z)` (yeni ölçümde), `predict_only(t)`
  (yan etkisiz ileri tahmin — lead noktası için).
- **NOT:** Hedef hız kanalı (`Target_Speed`) gerçek hızla korelasyonsuz
  bulunduğu için KULLANILMAZ; sadece konum ölçümüyle çalışılır.

### 3.2 Pürüzsüz Pursuit Güdümü — `guidance/pursuit_guidance.py`  ⭐ (aktif, v2)
Canlı modda görülen **dalgalı yol** sorununun kök nedenleri ve çözümleri:
- **`MeasurementGate`** — telemetri değeri **değişmedikçe** filtreye ölçüm
  beslemez (change-detection). Zamanlayıcı tabanlı `gps_counter` tamamen
  kaldırıldı. (Kök neden #1: bayat ölçüm.)
- **`LeadPointSmoother`** — lead noktasını LPF (tau) + slew-limit + `max_lag`
  ile yumuşatır; ölçüm anındaki sıçramayı ~tau saniyeye yayar. Slew tabanı
  **nominal** hız (filtre tahmini değil), böylece UKF hız tahmini geçici çökse
  bile aim gerçek lead'i takip eder.
- **`SmoothPursuitGuidance`** — Gate + Tracker + Smoother zinciri; `adaptive_lead`
  ile dönüş hızı arttıkça lead süresini kısaltır (viraj ekstrapolasyon hatasını
  sınırlar).
- **`YawController`** — P + **gyro damping** (dronun ölçülen yaw hızından
  sönümleme; hata türevinden DEĞİL). EMA yok. `kd=0.008` doğru; `kd=0.035`
  bang-bang üretir. Ağır EMA'nın faz gecikmesi/salınımı bu sayede giderildi.
- **Sonuç (10 tohum):** heading-rate RMS 31.8→19.1 deg/s, aim sıçraması p95
  18.5k→4.8k cm/s, yakalama süresi 18.8→17.7 s (10/10 başarılı).

### 3.3 Rota Tabanlı Intercept — `navigation/route_intercept_system.py`
Hedef tekrarlayan rota izlediğinden, her ölçümü bağımsız tahmin etmek yerine
**rotanın şeklini öğrenir**:
- `EllipseRouteModel` — Fitzgibbon direkt en-küçük-kareler elips fit +
  açısal hız (omega) ve yön (CW/CCW) kestirimi; `fit_robust()` outlier atar.
  Gerçek veride kalıntı std ~416 birim (yarı-eksenin ~%3.5'i), tur süresi ~30 s.
- `RoutePlanner` — canlı veriyle besler, sapma (residual) izler, eşik aşılırsa
  otomatik yeniden fit.
- `solve_intercept()` — dronun sabit hızıyla hedefi rota üzerinde en erken
  yakalayacağı zaman/nokta (lead pursuit çözümü).

**Canlı runner dayanıklılığı** (`tests/python/test_route_intercept.py` →
`LiveInterceptRunner`) — GPS kesintisi/stale-hold'a karşı sertleştirildi:
- **Faz dead-reckoning:** Rota kilitliyken hedefin elips faz açısı (`target_phi_est`)
  her döngüde `omega·dt` ile ilerletilir; SADECE gerçekten yeni ölçüm geldiğinde
  ölçülen faza doğru düzeltilir (`PHI_CORRECT_GAIN`). GPS **kesildiğinde (DROPOUT)**
  veya stale-hold tekrarında dron **yerinde saymaz** — hedefin muhtemel konumunu rota
  üzerinde tahmin edip `TRAIL_PHI` (~20°) arkasında kalır. Kestirim varken yaw
  dondurma da kaldırıldı (hareketli guide'a dönmeye devam eder).
- **Açık donma tespiti + kestirilen görüntü:** `stale_count` ile birkaç döngü
  (`STALE_HOLD_FRAMES`) boyunca değişmeyen veri veya kesinti "donmuş"
  (`target_frozen`) sayılır. Donunca hem mesafe hesabı hem ekrandaki hedef işareti
  ham (donmuş) konumu değil rota üzerindeki **kestirilen** hedef konumunu
  (`point_on_ellipse(target_phi_est)`) kullanır → hedef ekranda donuk görünmez,
  kestirdiği son konuma doğru ilerlemeye devam eder.
- **Fit yalnızca yeni ölçümle beslenir** (stale/dondurulmuş tekrarlar elips fitini tek
  noktada yoğunlaştırıp bozmaz); kesintideki `tz=0` irtifa ortalamasına girmez.
- **Reddetme→dondurma:** Tutarsız (yön/merkez) yeniden-fit reddedildiğinde bu, kilidin
  zaten kararlı olduğunun kanıtı sayılıp `orbit_count` artırılır → rota
  `LOCK_STABLE_LAPS` turdan sonra donar; eskiden görülen sonsuz "yeni fit reddedildi"
  spam'i (sayaç hiç artmadığı için rota hiç donmuyordu) biter.
- **Net tur tespiti:** Tur sayımı işaretli faz birikimiyle yapılır (`phase_acc += delta`,
  `abs` değil) → "tam tur" = başlangıç açısından itibaren NET 360°. Hedef 180°'de
  başlasa bile mutlak açının 360°'ye gelmesi tur saymaz; ölçüm titreşimi de erken
  tetiklemez.

### 3.5 Yarışma Modu İz Takibi (GPS-only) — 2026-11-18  ⭐ (aktif)
Şartnameye göre hedeften SADECE enlem/boylam/irtifa/yer hızı (≤5 Hz), avcıdan GPS +
skaler yer hızı + duruş (3.125–50 Hz) gelir. Hedef `rotation` kanalı ve dron hız vektörü
YOKTUR (sim gönderiyor ama kullanılamaz). Hat:
- `simulation/competition_telemetry.py` — sim telemetrisinden sadece izinli alanlar/hızlar.
- `navigation/gps_target_estimator.py` — koordineli dönüş modeli, 4 s robust Gauss-Newton,
  sıçramalar için segment offset, ileri tahminde dönüş hızı sönümü. Gecikme D=1.0 s parametre
  (konumdan gözlemlenemez).
- `navigation/hunter_state.py` — avcı hız vektörü KF (gönderilen komut + dron modeli + GPS + yer hızı).
- `guidance/follow_controller.py` — iz takibi (`est.trail(lag)` arayüzü), `simulation/follow_runner.py`
  varsayılan olarak bu modda çalışır (`--legacy` eski rotation'lı yol).
- Canlı (100 s): 40 m takip → mesafe 36–45 m, iz hatası medyan 3.5 m; 20 m takip dönüşlerde
  45–56 m'ye açılır (1 s GPS gecikmesi ≈ iz gecikmesi → ön görüş yok). Ayrıntı: `docs/yarisma_modu_takip.md`.

### 3.4 CTRV EKF (alternatif) — `navigation/ekf.py`
Config-driven (`config/ekf_params.yaml → ctrv_ekf`) 5 durumlu EKF
`[x, y, theta, v, omega]`, analitik Jacobian'lı. Mahalanobis spike reddi,
dropout tespiti, düşük Q + yüksek R ile pürüzsüz kestirim. UKF tracker'a
alternatif/karşılaştırma amaçlı (`ekf_deneme.py` deneme sürümü).

---

## 4. Kontrol / Uçuş Mantığı (`config/pid_params.yaml`)

- **İç PID** (attitude): pitch/roll/yaw stick komutları.
- **Dış PID** (outer): irtifa ve hız kontrolü.
- **Pursuit güdümü:** `follow_distance_cm = 500` (hedefin 5 m arkasında kal);
  imzalı mesafe hatası → hız düzeltmesi; `sqrt` hız profili
  (`v_safe = sqrt(2·approach_accel·|s|)`) ile simetrik yaklaşma/geçiş sınırı.
- **Seyir hızı** `chase_speed = 2800 cm/s` (~100 km/h, oyun içi max tırmanma
  limitiyle uyumlu).
- **SDK throttle semantiği:** -1 = serbest düşüş, 0 = hover, +1 = max tırmanış
  (dikey hız komutu; klasik gaz değil).

---

## 5. Teknoloji Yığını

| Alan | Teknoloji |
|---|---|
| Beyin | Python 3.8+, NumPy, filterpy (UKF), pandas, matplotlib |
| Görü | RF-DETR, ONNX, TensorRT, SAHI |
| GCS | Node.js, Electron, WebSocket (`ws`), Express, HTML/JS + Leaflet harita |
| Eski GUI | Tkinter (stdlib) |
| İletişim | Ham TCP soket, CSV satır protokolü, port 12345 |
| Veri | `hedef_konumlari.xlsx` (kalibrasyon/kayıt) |

---

## 6. Hızlı Başlangıç

```bash
# 1) Simülatörü başlat (DronesOfWar.exe) → TCP :12345 dinler
# 2) Python bağımlılıkları
pip install -r requirements.txt

# 3a) Eski Tkinter kontrol paneli + veri toplama
python drone_gui.py

# 3b) UKF tracker'ı kayıtlı veride görselleştir
python src/python/navigation/fixed_wing_tracker.py

# 3c) 1 Hz yarışma koşulu performans testi (UKF vs CV Kalman)
python src/python/navigation/simulate_1hz_test.py

# 3d) Canlı pursuit runner (simülatör bağlıyken)
python tests/python/test_pursuit_live.py

# 4) GCS web sunucusu (tarayıcı arayüzü + TCP köprü)
cd src/gcs && npm install && npm start   # HTTP :8080, WS :8765
```

---

## 7. Önemli Bağlam / Tuzaklar

- ⚠️ **Kök `README.md` kısmen eskimiştir** — silinmiş modüllere atıf yapar
  (`crsf_sender.py`, `gnss_filter.py`, `waypoint_controller.py`, `sim_bridge_ekf.py`
  vb.). Güncel gerçek için BU 3 dosyaya güven.
- ⚠️ Hedef GPS **stale hold** yapar → mutlaka `MeasurementGate` (değişim
  tespiti) kullan, zamanlayıcıyla besleme.
- ⚠️ Hedef **hız kanalına güvenme**; sadece konumdan hız kestir.
- ✅ `get_debug_truth()` yalnızca simülatörde debug açıkken gerçek (bozulmamış)
  değerleri verir — sadece test/karşılaştırma için, yarışmada kullanılmaz.
- 📌 Açık iş: canlı doğrulama ve terminal fazdaki taşma turları (drone hedeften
  hızlıyken yakın orbit) — bkz. `AI_GOREVLER.md`.
