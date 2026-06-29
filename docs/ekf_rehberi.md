# EKF Hedef Takip Filtresi — Kullanım ve Parametre Rehberi

**Dosya:** `src/python/navigation/ekf.py`  
**Config:** `config/ekf_params.yaml` → `target_ekf:`  
**Test:** `tests/python/navigation/live_ekf_monitor.py`

---

## 1. Neden EKF?

Yarışmada hedef uçağın GPS verisi şu sorunları içerebilir:

| Bozulma | SDK Bayrağı | Etki |
|---|---|---|
| Konum gürültüsü | `FLAG_NOISE` | ±100 cm seviyesinde rastgele sıçramalar |
| Ani zıplama | `FLAG_JUMP` | Binlerce cm'lik tek-çerçeve spike |
| Veri kesintisi | `FLAG_DROPOUT` | Konum saniyeler boyunca donar, hep aynı değer gelir |
| Sabit kayma | `FLAG_OFFSET` | Tüm ölçümler belirli bir yönde sistematik hata taşır |
| Gecikmeli veri | `FLAG_DELAY` | Gelen konum birkaç saniye öncesine ait |
| 1 Hz sınırı | `FLAG_RATELIMIT` | Saniyede yalnızca 1 güncelleme |

Ham GPS verisiyle çalışan bir guidance algoritması tüm bu durumlarda ya titrer ya da "donar". **Extended Kalman Filter (EKF)**, bu bozulmaları süzerek her döngüde güvenilir, akıcı bir konum tahmini üretir.

---

## 2. Durum Vektörü ve Model

### 2.1 Durum Vektörü

EKF 6 boyutlu bir durum vektörü tutar:

```
x = [px, py, pz, vx, vy, vz]
```

- `(px, py, pz)` — konum (cm, Unreal Engine dünya koordinatları)
- `(vx, vy, vz)` — hız (cm/s)

### 2.2 Süreç Modeli — Sabit Hızlı Hareket

Her `predict` adımında:

```
px_yeni = px + vx · dt
py_yeni = py + vy · dt
pz_yeni = pz + vz · dt
vx_yeni = vx   (sabit, ancak süreç gürültüsü yön değişimine izin verir)
vy_yeni = vy
vz_yeni = vz
```

**Sabit hız kısıtı:** Hedef uçak her zaman 1500 cm/s hızla gider. Her güncelleme adımı sonunda hız vektörü normalize edilir:

```
v → v / |v| × 1500 cm/s
```

Bu sayede filtre yön değişimlerini takip edebilirken hız büyüklüğü her zaman gerçek değerde kalır. Hedef sabit bir rotayı gidip geliyorsa hız vektörü yönü doğal olarak güncellenir.

### 2.3 Ölçüm Modeli

Yalnızca konum ölçülür (GPS):

```
z = [px_ölçülen, py_ölçülen, pz_ölçülen]
H = [[1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0]]
```

---

## 3. Bozulma Korumaları

### 3.1 Dropout Tespiti (Dondurulmuş Ölçüm)

Ardışık iki ölçüm arasındaki mesafe `min_position_change_cm`'den küçükse ölçüm "dondurulmuş" kabul edilir.

```
|z_yeni − z_önceki| < min_position_change_cm  →  sadece predict, güncelleme yok
```

Gerçek bir 1500 cm/s hızla giden uçak 1 saniyede 1500 cm ilerler; ölçüm değişmiyorsa kesinlikle dropout vardır.

### 3.2 Spike Reddi — Mahalanobis Mesafesi

Gelen ölçüm, filtrenin öngördüğü konumdan çok uzaksa yoksayılır:

```
inovasyon:       y = z − H · x_tahmin
inovasyon kov.:  S = H · P · Hᵀ + R
Mahalanobis:     d = √(yᵀ · S⁻¹ · y)

d > mahalanobis_threshold  →  spike! güncelleme atlanır
```

`d` birimi σ (sigma) olduğundan threshold'u sezgisel ayarlamak kolaydır: `5.0` değeri istatistiksel olarak beklenen ölçümlerin %99.99'unu geçirir, yalnızca gerçek sıçramaları reddeder.

### 3.3 Bootstrap (Başlangıç)

EKF, `startup_samples` kadar ölçüm toplandıktan sonra aktif olur. Bu sürede ham GPS döndürülür. İlk iki ölçümden başlangıç hız yönü tahmin edilir.

---

## 4. Config Parametreleri

**Dosya:** `config/ekf_params.yaml` → `target_ekf:`

```yaml
target_ekf:
  constant_speed_cms: 1500.0      # Hedef uçağın sabit hızı
  
  process_noise:
    position: 500.0               # Konum süreç gürültüsü (cm²)
    velocity: 80000.0             # Hız yön gürültüsü ((cm/s)²)
  
  measurement_noise:
    position: 10000.0             # GPS ölçüm gürültüsü (cm²) ≈ 100 cm std
  
  initial_covariance:
    position: 1000000.0           # Başlangıç konum belirsizliği (cm²)
    velocity: 4000000.0           # Başlangıç hız belirsizliği ((cm/s)²)
  
  outlier_rejection:
    mahalanobis_threshold: 5.0    # Spike red eşiği (σ)
    min_position_change_cm: 10.0  # Dropout tespit eşiği (cm)
  
  init:
    startup_samples: 2            # Aktifleşme için gereken ölçüm sayısı
```

### Parametre Etkileri

| Parametre | Küçük değer | Büyük değer |
|---|---|---|
| `process_noise.velocity` | Yön değişimlerine yavaş tepki | Hızlı adaptasyon ama daha sallantılı tahmin |
| `measurement_noise.position` | GPS'e daha fazla güven (gürültüde bozulur) | Filtreye daha fazla güven (gerçek hareketi geciktirir) |
| `mahalanobis_threshold` | Çok katı → normal ölçümler de reddedilir | Çok gevşek → spike'lar filtreden geçer |
| `min_position_change_cm` | Çok küçük → dropout'ları kaçırır | Çok büyük → yavaş hareketleri dropout sanır |

---

## 5. API Kullanımı

```python
from src.python.navigation.ekf import TargetEKF

ekf = TargetEKF()  # config/ekf_params.yaml otomatik yüklenir

# Her döngüde:
raw_pos = drone.get_target_location()   # (x, y, z) cm
est_pos = ekf.update(raw_pos)           # filtrelenmiş (x, y, z)

# İleri tahmin (lead angle için):
predicted = ekf.predict(lookahead_s=0.5)  # 0.5 saniye sonra nerede olacak

# Hız yönü (pursuit için):
vel = ekf.get_estimated_velocity()   # (vx, vy, vz) cm/s

# Durum kontrolü:
if not ekf.is_ready:
    print("Henüz bootstrap bitmedi, ham veri kullanılıyor")

# Tanılama:
print(ekf.last_maha)             # Son Mahalanobis mesafesi
print(ekf.last_spike_rejected)   # Spike reddedildi mi?
print(ekf.last_dropout)          # Dropout tespit edildi mi?
```

---

## 6. Canlı Test — `live_ekf_monitor.py`

```bash
python3 tests/python/navigation/live_ekf_monitor.py
python3 tests/python/navigation/live_ekf_monitor.py --no-plot   # sadece terminal
python3 tests/python/navigation/live_ekf_monitor.py --port 12345
```

Ekran 4 panel gösterir:

| Panel | İçerik |
|---|---|
| Sol üst | XY yörüngesi — ham GPS (gri), EKF (cyan), ground truth (lime) |
| Sağ üst | Irtifa (Z) zaman serisi |
| Sol alt | Konum hatası vs ground truth (debug modda) |
| Sağ alt | Mahalanobis mesafesi — kırmızı çizgi = spike eşiği |

Kapanışta RMSE ve iyileştirme yüzdesi yazdırılır.

---

## 7. Veri Akışı

```
drone_sdk.get_target_location()
        │
        ▼
  dropout mu? ─── Evet ──► predict only → konum tahmini (hız ile ilerler)
        │
       Hayır
        │
        ▼
  Mahalanobis > 5σ? ─── Evet ──► spike! predict only → önceki tahmin korunur
        │
       Hayır
        │
        ▼
  EKF predict + update → hız normalize (1500 cm/s) → filtrelenmiş konum
```

---

## 8. Önemli Notlar

- EKF Unreal Engine **santimetre** koordinat sisteminde çalışır; metre'ye çeviri `/ 100` ile yapılır.
- `predict(lookahead_s)` fonksiyonu guidance döngüsünde hedefin biraz ilerisini hedeflemek (lead angle) için kullanılabilir.
- Dropout süresince konum tahmini hız vektörü ile ilerler; uzun dropout'larda belirsizlik büyür (P matrisi şişer), dropout bitince hızla güncellenir.
- Sabit hız kısıtı (`_enforce_speed_constraint`) her güncelleme sonrası otomatik uygulanır; manuel çağrı gerekmez.
