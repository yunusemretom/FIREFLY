# Sim Bridge EKF — Agresif Takip Köprüsü Rehberi

**Dosyalar:**  
- `tests/python/sim_bridge_ekf.py` — agresif kovalama bridge'i  
- `tests/python/sim_bridge_pursuit.py` — 5 m arkadan takip bridge'i  
- Referans (değişmez): `src/python/simulation/sim_bridge.py`

---

## 1. Orijinal `sim_bridge.py`'den Farklar

Orijinal dosyaya hiç dokunulmamıştır. Test dosyaları üç sorunu çözmek için yazılmıştır:

| Sorun (Orijinal) | Çözüm (Test Dosyaları) |
|---|---|
| **KALKIŞ** bekleme: pitch=0 tırmanıyor, hareketsiz | Pitch + throttle **her zaman eş zamanlı** çalışır |
| **yaw_scale** = cos(90°) = 0 → büyük açılarda drone donar | `max(0.3, cos(yaw_error))` → her zaman min %30 ileri |
| Hedef GPS: 1 Hz, gürültülü, spike/dropout var | **Truth > EKF > Ham GPS** öncelik sırası |
| TAKIP modunda hız SEYIR'den düşük (donma hissi) | Mesafe bazlı sürekli agresif hız |

---

## 2. Konum Kaynağı Öncelik Sırası

Her iki test bridge'i de aynı `_best_target_pos()` mantığını kullanır:

```
1. truth_data["available"] == True
   → Sunucu debug modunda → truth konum kullan
   → EKF'yi de truth ile besle (ısınmış tut, hız tahmini güncel kalsın)

2. ekf.is_ready == True
   → EKF aktif → filtrelenmiş konum kullan

3. Hiçbiri hazır değil
   → Ham GPS döndür (yalnızca ilk 2 ölçüm toplanırken)
```

### Neden Truth Önce?

Truth verisi sunucunun **drone'un tam döngü Hz'inde** (50 Hz) güncellediği bozulmamış konumdur. 1 Hz GPS'in donduğu durumlarda bile truth her frame'de yeni değer taşır → guidance döngüsü hiç donmaz.

Truth yoksa (yarışma modu) EKF devreye girer ve 1 Hz ölçümler arası hız modeliyle ara değerleri tahmin eder.

---

## 3. `sim_bridge_ekf.py` — Agresif Kovalama

**Amaç:** Hedefe mümkün olan en hızlı şekilde yaklaşmak ve mesafeyi kapatmak.

### Durum Mantığı

```
dist_3d > takip_dist (8000 cm)  →  SEYIR: chase_speed ile git (2800 cm/s = 100 km/h)
dist_3d ≤ takip_dist            →  TAKIP: hedef_hızı + marj (1500 + 500 = 2000 cm/s)
```

Bu iki durum arası geçiş `ModeManager` kullanmadan doğrudan hesaplanır; KALKIŞ durumu yoktur.

### Pitch Hesabı (Inline)

```python
speed_error_ms  = (target_speed_cmd − drone_speed) / 100.0
speed_adj       = speed_pid.calculate(speed_error_ms, dt)

camera_center   = target_elevation − camera_tilt
raw_pitch       = clamp(camera_center − speed_adj, min_pitch, max_pitch)
smooth_pitch    = slew.update(raw_pitch, dt)

yaw_scale       = max(0.3, cos(yaw_error))   # minimum %30
pitch_cmd       = pitch_normalized × yaw_scale
```

`ModeManager` ve `PitchController` sınıfları yerine doğrudan `PID` ve `SlewLimiter` kullanılmasının sebebi: KALKIŞ durumunu ("state == KALKIS → pitch = 0") bypass etmek.

### Terminal Çıktısı

```
[SEYIR|TRUTH] dist= 15400 dz=  -90 hız= 48.2km/h hedef=101km/h | yaw=+11.3° pit=-32.1° thr=+0.07
[TAKIP|EKF]   dist=  5200 dz=   +5 hız= 97.3km/h hedef= 72km/h | yaw= -1.8° pit=-18.0° thr=+0.01
```

| Alan | Açıklama |
|---|---|
| `[MOD\|KAYNAK]` | SEYIR/TAKIP + konum kaynağı |
| `dist` | 3D mesafe (cm) |
| `dz` | İrtifa farkı (+ = hedef yukarıda) |
| `hız` | Drone'un mevcut hızı (km/h) |
| `hedef` | Guidance'ın hedeflediği hız (km/h) |
| `yaw°` | Hedefe açı hatası |
| `pit°` | Pitch açısı |
| `thr` | Throttle komutu |

---

## 4. `sim_bridge_pursuit.py` — 5 m Arkadan Takip

**Amaç:** Hedefe tam `follow_distance_cm` mesafede kalmak, ne geçmek ne geri kalmak.

Detaylı açıklama için: **[pursuit_rehberi.md](pursuit_rehberi.md)**

### Önemli Fark

`sim_bridge_ekf.py`, `PitchController` sınıfını bypass eder ve pitch'i inline hesaplar.  
`sim_bridge_pursuit.py`, tüm eksenleri `PursuitController` sınıfına devreder:

```python
cmds, telem = pursuit.calculate(
    drone_pos, drone_rot, drone_speed,
    target_pos, target_vel, dt
)
throttle, pitch, roll, yaw = cmds
```

### EKF Hız Vektörü Gereksinimi

Pursuit, "takip noktası" hesabı için hedefin **hız yönünü** bilmek zorundadır. Bu bilgi EKF'den gelir:

```python
target_vel = ekf.get_estimated_velocity()   # (vx, vy, vz) ≈ 1500 cm/s büyüklüğünde
```

EKF henüz hazır değilse `(1500, 0, 0)` varsayılan kullanılır ve drone düz ileri gider.

---

## 5. Hangi Bridge Ne Zaman Kullanılır?

| Bridge | Kullanım Senaryosu |
|---|---|
| `sim_bridge.py` (orijinal) | Temel referans, dokunulmaz |
| `sim_bridge_ekf.py` | Hedefe maksimum hızda yaklaşma, mesafe kapatma testi |
| `sim_bridge_pursuit.py` | 5 m (veya config'den ayarlanan mesafede) hassas takip |

### Koşturma

```bash
# Her ikisi de proje kök klasöründen çalıştırılır:
python3 tests/python/sim_bridge_ekf.py
python3 tests/python/sim_bridge_pursuit.py
```

---

## 6. Config Parametreleri

Tüm parametreler `config/pid_params.yaml`'dan okunur. Test bridge'leri için ilgili bölümler:

```yaml
flight:
  camera_tilt: 20.0           # Kamera eğim açısı (°) — pitch denklemini etkiler
  chase_speed: 2800.0         # sim_bridge_ekf SEYIR modu hedef hızı (cm/s)
  takip_speed_margin: 500.0   # sim_bridge_ekf TAKIP modu ekstra hız (cm/s)
  dt: 0.02                    # Döngü süresi (50 Hz)

limits:
  min_pitch: -60.0            # En dik öne eğim (°)
  max_pitch: 20.0             # En dik geri eğim (°)

thresholds:
  takip_dist: 8000.0          # sim_bridge_ekf: SEYIR→TAKIP geçiş mesafesi (cm)

pursuit:
  follow_distance_cm: 1000.0  # sim_bridge_pursuit: takip mesafesi (cm)
  # (diğer parametreler için bkz. pursuit_rehberi.md)
```

---

## 7. Geliştirme Notları

- **camera_tilt ayarı:** Bu değer pitch denkleminde `camera_center = target_elev − camera_tilt` olarak kullanılır. Kamera yere daha dik bakıyorsa artırın; yatay bakıyorsa azaltın. Yanlış değer drone'un hedefe açısını hesaplarken kaymasına yol açar.

- **yaw_scale minimum değeri (%30):** Bu değer çok düşürülürse (örn. 0.1) drone büyük açılarda neredeyse durur. Çok yüksek tutulursa (örn. 0.6) drone geniş yay çizerek hedefe ulaşır. 0.3 dengeli bir değerdir.

- **EKF ısınma süresi:** Başlangıçta 2 ölçüm toplanana kadar (`startup_samples`) ham GPS döner. Bu süre ~2 saniyedir (1 Hz GPS). Bu sürede drone ileri gider ama yön tahmini doğru değildir; pursuit için EKF'nin ısınmasını beklemek önerilir.
