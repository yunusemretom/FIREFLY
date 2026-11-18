# Yarışma Modu İz Takibi (GPS-only)

> Durum: 2026-11-18 — canlı simülatörde doğrulandı.
> Kod: `src/python/simulation/competition_telemetry.py`, `src/python/navigation/gps_target_estimator.py`,
> `src/python/navigation/hunter_state.py`, `src/python/guidance/follow_controller.py`,
> `src/python/simulation/follow_runner.py`

## 1. Neden yeniden yazıldı?

Şartnamenin "Haberleşme paketi ve limit bilgileri" bölümü takımlara verilecek telemetriyi
sınırlıyor:

| Araç | Gelen alanlar | Hız |
|---|---|---|
| Hedef İHA | enlem, boylam, irtifa (kalkış noktasına göre), yer hızı | en fazla **5 Hz** |
| Avcı dron | GPS konum (LLA), **skaler** yer hızı, duruş (roll, pitch, yaw) | TX ayarına bağlı: **3.125 Hz** (1:32) … **50 Hz** (1:2) |

Angle mod limitleri: yatış 60°, dikilme 60°.

Simülatör ise bunlardan fazlasını gönderiyor: hedefin yönelim (`rotation`) kanalı (~100 Hz,
gecikmesiz), dronun hız **vektörü** ve her şey ~100 Hz. 2026-11-17'deki ilk takip sistemi
(`talon_estimator.py`) tam olarak bu fazladan rotation kanalına dayanıyordu, yani yarışmada
çalışmazdı. Bu doküman, sadece şartnamedeki verilerle çalışan yeni hattı anlatır.

## 2. Veri akışı

```
drone_sdk (sim, ~100 Hz, fazla kanallı)
        │
        ▼
CompetitionTelemetry ── sadece şartname alanları, şartname hızlarında
        │                      │
        │ hedef konum (≤5 Hz)  │ avcı GPS + yer hızı + duruş (3.125–50 Hz)
        ▼                      ▼
GpsTargetEstimator       HunterStateEstimator ◄── gönderilen stick komutları
  (konum, hız, iz)         (konum, hız vektörü)
        │                      │
        └──────────┬───────────┘
                   ▼
        TrailFollowController ── throttle / pitch / roll / yaw
                   │
                   ▼
           drone_sdk.set_control_surfaces (tek TCP satırı, 50 Hz)
```

Kontrol yolu telemetriyi **sadece** `CompetitionTelemetry` üzerinden okur. `get_debug_truth()`
yalnızca metrik ve grafik içindir, kontrole girmez.

## 3. Bileşenler

### 3.1 `CompetitionTelemetry` — şartname filtresi
- Avcı: `position`, `ground_speed = hypot(vx, vy)`, `rotation`. `drone_hz` ile örnekle-tut
  (sample-and-hold).
- Hedef: `position`, `speed`. Sadece değer değiştiğinde ve en fazla 5 Hz.
- Her paket `stamp` ve `fresh` bayrağı taşır.
- Yerel koordinatlar cm cinsinden (X, Y, Z). Yarışmada LLA gelecek; LLA → yerel ENU
  dönüşümü bu katmanın girişine eklenecek (açık iş).

### 3.2 `GpsTargetEstimator` — sadece konumdan hedef kestirimi

**Model** (yatayda sabit hızlı koordineli dönüş, dikeyde sabit dikey hız):

```
ψ(s) = ψ0 + ω·s
x(s) = x0 + V/ω · (sin(ψ0+ωs) − sin ψ0)
y(s) = y0 + V/ω · (cos ψ0 − cos(ψ0+ωs))
z(s) = z0 + vz·s
```

- **Gecikme:** `t_k` anında alınan konum, hedefin `t_k − D` anındaki konumudur. Sim'de
  ölçülen değer D = 1.0 s. D konum verisinden tek başına gözlemlenemez (sabit gecikme,
  iz boyunca geride olmaya eşdeğer), bu yüzden parametre olarak verilir (`set_delay`).
- **Çözüm:** Son 4 s'lik ölçümlere robust Gauss-Newton (5 parametre). Önceki çözümden
  sıcak başlangıç + doğru uydurma adayı denenir, maliyeti düşük olan seçilir.
- **Sıçramalar — segment modeli:** İki ardışık ölçüm arasında fiziksel olarak imkânsız bir
  adım (`step_cm` = 10 m sapma) yeni bir segment başlatır. Her segmentin kendi sabit
  offset'i oturtmada ayrıca çözülür. Böylece sıçrama sırasındaki noktalar atılmaz, dönüş ve
  hız bilgisini taşımaya devam eder. Yeni segmentin ilk noktası eski bir seviyeye uyuyorsa
  (sıçrama bitti) o segmente katılır. Referans çerçeve, son 5 s'te en çok noktası olan
  segmenttir. Kalıcı bir offset değişimi zamanla ana segment olur.
- **Kesinti:** Ölçüm gelmezken model ileri tahmin eder. Son ölçümden sonra dönüş hızı
  0.8 s zaman sabitiyle sıfıra söner (yanlış bir ω'yı saniyelerce uzatmak, düz uçuş
  varsaymaktan daha kötüdür).
- **Çıktılar:** `position()`, `velocity()`, `predict(lead)`, `trail(lag)` (hedefin `lag` s
  önceki konum/hız/ivmesi), `state()`. `extra_offset` dışarıdan bir offset düzeltmesi
  eklemek içindir.

Önemli parametreler (`GpsTargetEstimator.__init__`):

| Parametre | Varsayılan | Anlamı |
|---|---|---|
| `delay_s` | 1.0 | Hedef konum gecikmesi |
| `window_s` | 4.0 | Oturtma penceresi |
| `outlier_cm` | 800 | Pencere içi aykırı eşiği |
| `step_cm` / `merge_cm` | 1000 / 1000 | Segment başlatma / birleştirme |
| `main_window_s` | 5.0 | Ana segment çoğunluk penceresi |
| `turn_decay_s` | 0.8 | İleri tahminde dönüş hızı sönümü |
| `use_speed_channel` | False | Hız kanalını V ölçümü olarak kullan (sim'de çöp, kapalı) |

### 3.3 `HunterStateEstimator` — avcının hız vektörü
Yarışmada avcının hız vektörü gelmez. Kalman filtresi, durum `[px, py, vx, vy]` (+ ayrı
`[z, vz]`):
- **Tahmin:** Ölçülen dron modeli + **bizim gönderdiğimiz** stick komutları:
  eğim = 60°·stick (0.15 s gecikmeyle izler), `dv/dt = (151·eğim − v)/τ`, τ = 2 s.
- **Güncelleme:** GPS konumu (doğrusal) + yer hızı büyüklüğü (EKF).
- Komutlar bilindiği için seyrek GPS arasında hız doğru ilerler. Canlıda 3.125 Hz ile de
  sorunsuz çalıştı.

### 3.4 `TrailFollowController` — iz takibi
Hedefin **kendi izi** üzerinde, `follow_distance` kadar geride, onunla aynı hızda uçar:

```
p_ref = hedefin (follow_distance / V) s önceki konumu   (est.trail)
v_des = v_ref + kapanma · birim(p_ref − p_dron)
kapanma = min(Kp·|e|, √(2·a_fren·|e|), v_max)
```

Fren profilli kapanma, referansa göreli hız sıfırla varılmasını sağlar; aşım ve tur atma
olmaz. Hız → eğim dönüşümü dron modelinin tersidir (`(v_des + τ·a_cmd)/151`), dönüşlerde
iz noktasının merkezcil ivmesi feed-forward olarak eklenir. Yaw burnu (kamerayı) hedefe
tutar. Modlar: YAKLAŞMA, İZ, DOLANMA (dron hedefin önündeyse yandan dolanır), KAÇINMA
(çarpışma önleme yarıçapı).

## 4. Sonuçlar

**Kestirici, çevrimdışı** (`analysis/replay_gps_estimator.py`, 10 kayıt, sim çökmesinden
sonraki donmuş kısımlar atılarak):

| Ölçüt | Medyan | p95 | Maks |
|---|---|---|---|
| Anlık konum hatası | 4.6 m | 17.6 m | 78 m |
| İz noktası hatası (1.1 s geride) | 3.0 m | 8.0 m | 52 m |

İz hatasının ~3 m'lik kısmı GPS'in sabit offset'idir; sadece GPS ile gözlemlenemez.

**Canlı simülatör, 100 s** (`follow_runner`, t > 15 s, debug truth ile ölçüm):

| Mod | Mesafe medyan / p95 | İz hatası medyan / p95 | Hedefi geçme |
|---|---|---|---|
| 20 m takip, avcı 50 Hz | 25 m / 56 m | 8.8 m / 38 m | 0 |
| **40 m takip, avcı 50 Hz** | **39 m / 45 m** | **3.5 m / 9.4 m** | 0 |
| 40 m takip, avcı 3.125 Hz | 39 m / 47 m | 3.7 m / 10.4 m | 0 |

**20 m neden kötü:** GPS gecikmesi (1 s) ≈ 20 m'deki iz gecikmesi (1.1 s). Yolu ancak iz
noktasına kadar biliyoruz, ilerisini göremiyoruz. τ ≈ 2 s ataletli dron dönüşlere geç kalıp
15–25 m açılıyor. 40 m'de ~1 s ön görüş oluşuyor.

## 5. Bilinen sınırlar
- **D sabit:** Yarışmadaki gerçek gecikme farklıysa iz bu fark kadar kayar.
- **En zor durum:** Kesinti + sıçrama + dönüş aynı anda gelirse birkaç saniyelik 25–30 m
  hata oluşur.
- **Avcı kestiricisi** çevrimdışı test edilemiyor (simüle edilen dron ile aynı modeli
  kullanıyor). Doğrulama sadece canlı testlerden geliyor.
- **Kamera:** Mevcut RF-DETR modeli Talon'u arkadan sadece ~10–15 m'de güvenilir buluyor.
  30–40 m'de neredeyse kör (bkz. `simulator_ve_ortam.md`).

## 6. Çalıştırma

```bash
# Simülatörü aç (açıksa ve telemetri akıyorsa dokunmaz)
scripts/sim_up.sh

# Canlı yarışma modu (varsayılan), 40 m takip, 100 s
python3 -m src.python.simulation.follow_runner --seconds 100 --follow 4000

# En kötü avcı telemetri hızıyla
python3 -m src.python.simulation.follow_runner --seconds 100 --follow 4000 --drone-hz 3.125

# Eski rotation'lı yol (yarışmada YOK, sadece karşılaştırma)
python3 -m src.python.simulation.follow_runner --legacy

# Simülatörsüz kapalı döngü (kayıt + dron modeli)
python3 tests/python/test_follow_offline.py --follow 4000
python3 tests/python/test_follow_offline.py --drone-hz 3.125

# Kestiriciyi kayıtlarda değerlendir
python3 analysis/replay_gps_estimator.py analysis/data/rec1.csv
```

Canlı koşu `analysis/data/<out>.csv` ve `.png` üretir (`--out` ile değiştirilebilir).
