# Pursuit Güdüm Kontrolcüsü — Kullanım ve Parametre Rehberi

**Dosya:** `src/python/guidance/pursuit.py`  
**Config:** `config/pid_params.yaml` → `pursuit:`  
**Bridge:** `tests/python/sim_bridge_pursuit.py`

---

## 1. Ne Yapar?

`PursuitController`, dronu hedef uçağın tam **`follow_distance_cm`** (varsayılan 1000 cm = 10 m) **arkasında** sabit tutmak için tasarlanmıştır.

Temel davranışlar:
- Geri kalınca → hızlanır
- Doğru konumda → hedefle aynı hızda gider
- Çok yaklaşınca → ileri hızını kısar, **kendi etrafında dönmez**
- Dönüş sırasında → %30 minimum ileri hareketi korur, dönüş tamamlanınca tam hıza geçer

---

## 2. Matematiksel Temel

### 2.1 Takip Noktası (Follow Point)

EKF'den gelen hız vektörünün normalize edilmiş yönü `û` kullanılarak sanal bir "takip noktası" hesaplanır:

```
û = target_vel / |target_vel|          (birim yön vektörü)

follow_point = target_pos − û × follow_distance_cm
```

Bu nokta hedefin tam `follow_distance_cm` gerisindedir ve hedef hareket ettikçe otomatik güncellenir.

### 2.2 İmzalı Mesafe (s)

Drone'un takip noktasına göre konumu tek bir sayıyla ifade edilir:

```
r = drone_pos − follow_point
s = r · û   (nokta çarpımı, izdüşüm)
```

| s değeri | Anlam | Yapılacak |
|---|---|---|
| `s < 0` | Drone geri kaldı | Hızlan |
| `s = 0` | Mükemmel konum | Hedef hızını koru |
| `s > 0` | Drone çok yakın | Yavaşla |
| `s >> 0` | Drone geçmek üzere | Önemli ölçüde yavaşla |

### 2.3 Hız Düzenlemesi — Dış PID + Sqrt Profili

Sistemin iki katmanlı hız kontrolü vardır:

**Adım 1 — Dış PID:**
```
hata = 0 − s = −s
raw_correction = dist_pid.calculate(hata, dt)
```

**Adım 2 — Sqrt hız profili (overshoot güvencesi):**
```
v_safe = sqrt(2 · approach_accel · |s|)       [sadece s < 0 iken]
correction = min(raw_correction, v_safe)

target_speed = hedef_hızı + correction
target_speed = clamp(target_speed, min_speed, max_speed)
```

Fiziksel gerçek: `approach_accel` ivmeyle frenleme yapan bir araç, `d` mesafede en fazla `v_safe` hızında yaklaşabilir. Bu limiti aşarsa kalan mesafede duramaz → overshoot.

Örnek sayısal değerler (`kp=0.4`, `approach_accel=300`):

| s (cm) | P terimi | sqrt_cap | Uygulanan | Hedef hız |
|---|---|---|---|---|
| −2000 | 800 | 1095 | **800** (P) | 2300 |
| −1000 | 400 | 775 | **400** (P) | 1900 |
| −500 | 200 | 548 | **200** (P) | 1700 |
| −200 | 80 | 346 | **80** (P) | 1580 |
| −100 | 40 | 245 | **40** (P) | 1540 |
| 0 | 0 | — | **0** | 1500 |
| +200 | −80 | — | **−80** | 1420 |

> **Neden `kd=0`?** D terimi, mesafe kapanırken negatif türev üretir ve uzaktan yaklaşmayı fazlaca frenler. Sqrt profili frenleme görevini zaten fiziksel olarak garanti eder; D terimi gereksiz ve zararlıdır.

**I terimi** küçük sabit kaymaları ortadan kaldırır (örn. sürekli bir disturbance).

### 2.4 İşaret Değişiminde Integral Sıfırlama

`s` işareti değiştiğinde (drone follow_point'i geçip geri döndüğünde) integral birikim sıfırlanır:

```python
if prev_s * s < 0:
    dist_pid.integral = 0.0
```

Bu, uzun süre geride kalan drone'da biriken integralin geçiş noktasında overshoot yapmasını engeller (anti-windup).

---

## 3. Spin Önleme — Yaw Tasarımı

### Eski yaklaşım (spin eden)

```
# Yaw → follow_point'e bak
dx = follow_point.x − drone.x
```

`s > 0` olduğunda follow_point drone'un **arkasına** düşer → bearing 180° ters döner → yaw kontrolörü dronu tam çevirir → **spin**.

### Yeni yaklaşım (spin yok)

```
# Yaw → her zaman HEDEFE bak
tdx = target_pos.x − drone.x
```

Hedef her zaman drone'un önündedir (normal takip senaryosunda). Drone yavaşlayıp hedefin uzaklaşmasını bekler; geriye dönme ihtiyacı hiç doğmaz.

### Dönüş Sırasında İleri Hareketi

```python
yaw_scale = max(0.3, cos(yaw_error))
pitch_cmd = pitch_cmd × yaw_scale
```

| Yaw hatası | Ölçek | Drone davranışı |
|---|---|---|
| 0° | 1.00 | Tam hız ileri |
| 30° | 0.87 | %87 ileri + döner |
| 60° | 0.50 | %50 ileri + döner |
| 90° | 0.30 | %30 ileri + döner (minimum) |
| 120° | 0.30 | %30 ileri + döner (minimum) |

Minimum %30 sayesinde drone büyük açı farkı olan dönüşlerde bile **tamamen durmaz**.

---

## 4. Dikey Eksen — Throttle

Throttle, hedef uçağın **irtifasına** eşleşmek için çalışır:

```
dz = target_pos.z − drone.z       (cm)
alt_corr = alt_pid.calculate(dz / 100, dt)   (metre cinsinden)
throttle = base_throttle + alt_corr
```

Bu, pitch (ileri hız) ve throttle (irtifa) eksenlerinin birbirinden bağımsız çalışmasını sağlar. Drone hem irtifa ayarlarken hem ilerler; "önce tırman, sonra git" davranışı yoktur.

---

## 5. Config Parametreleri

**Dosya:** `config/pid_params.yaml` → `pursuit:`

```yaml
pursuit:
  follow_distance_cm: 1000.0   # Hedefin kaç cm arkasında kal
                                # 50   →  0.5 m (çok yakın)
                                # 500  →  5 m
                                # 1000 → 10 m (tipik)
                                # 2000 → 20 m

  distance_pid:                 # Dış döngü: mesafe → hız
    kp: 0.4                     # Orantısal: 100 cm hata → 40 cm/s düzeltme
    ki: 0.02                    # İntegral: sabit kayma giderir
    kd: 0.0                     # Türev KULLANILMAZ — sqrt profili frenlemeyi üstlenir
    int_limit: 100.0            # İntegral sınırı ±100 cm/s

  approach_accel_cms2: 300.0    # Sqrt profil fren bütçesi
                                # v_safe = sqrt(2 * 300 * |s|)
                                # Küçük → yavaş/yumuşak, Büyük → hızlı/agresif

  min_speed_cms: 200.0          # Minimum ileri hız (durma engeli)
  max_speed_cms: 2800.0         # Maksimum yaklaşma hızı (~100 km/h)
```

### Parametre Etkileri

| Parametre | Küçük değer | Büyük değer |
|---|---|---|
| `follow_distance_cm` | Daha yakın takip, overshoot riski artar | Daha geniş marj, daha az hassas |
| `distance_pid.kp` | Yavaş yaklaşım (uzun süre geri kalır) | Hızlı yaklaşım ama salınım riski |
| `distance_pid.ki` | Sabit kayma kalabilir | Kayma giderilir, windup riski artar |
| `distance_pid.kd` | (0 tutulmalı — sqrt profili var) | Uzaktan yaklaşmayı aşırı frenler |
| `approach_accel_cms2` | Yavaş/yumuşak yaklaşım, az overshoot | Hızlı kapanma ama geç frenleme |
| `min_speed_cms` | Drone durabilir/geri gidebilir | Drone her zaman biraz ilerler |
| `max_speed_cms` | Yaklaşma yavaşlar | Daha hızlı kapanma |

### Hızlı Ayarlama Kılavuzu

**Drone salınım yapıyor (çok yaklaşıp uzaklaşıyor):**
→ `kp` düşür (0.4→0.2) veya `approach_accel_cms2` düşür

**Drone geri kalıyor, yetişemiyor:**
→ `kp` artır (0.4→0.6) veya `max_speed_cms` artır

**Drone yavaş yavaş yaklaşıp uzaklaşıyor (drift):**
→ `ki` küçük artır (0.02→0.05)

**Mesafe doğruyken hız çok fazla/az:**
→ İç hız döngüsünü ayarla: `outer_pid.speed.kp` (`pid_params.yaml`)

---

## 6. Kullanım

### 6.1 Doğrudan Sınıf Kullanımı

```python
from src.python.guidance.pursuit import PursuitController
from src.python.navigation.ekf import TargetEKF
import yaml

with open("config/pid_params.yaml") as f:
    config = yaml.safe_load(f)

pursuit = PursuitController(config)
ekf     = TargetEKF()

# Her döngüde:
raw_pos    = drone.get_target_location()
target_pos = ekf.update(raw_pos)
target_vel = ekf.get_estimated_velocity()   # 1500 cm/s normalize

cmds, telem = pursuit.calculate(
    drone_pos    = drone.get_drone_location(),
    drone_rot    = drone.get_drone_rotation(),
    drone_speed  = drone.get_drone_speed(),
    target_pos   = target_pos,
    target_vel   = target_vel,
    dt           = 0.02,
)
throttle, pitch, roll, yaw = cmds
drone.set_control_surfaces(throttle, -pitch, -roll, yaw, True)
```

### 6.2 Hazır Bridge ile

```bash
# Proje kök klasöründen:
python3 tests/python/sim_bridge_pursuit.py
```

---

## 7. Terminal Çıktısı

```
  TRUTH            -30     54.0k    53.8k   +1.2   -18.4  +0.05  OK
  TRUTH           +210     50.4k    54.2k   +0.5   -17.9  +0.00  YAKIN
  EKF[DROPOUT]    -180     56.0k    52.1k   +0.9   -19.1  +0.08  GERİ
```

| Sütun | Açıklama |
|---|---|
| Kaynak | `TRUTH` / `EKF` / `EKF[SPIKE]` / `EKF[DROPOUT]` / `RAW` |
| s (cm) | İmzalı mesafe: negatif=geri, pozitif=çok yakın |
| HedefHz | Pursuit'in drone'a verdiği hedef hız (km/h) |
| DroneHz | Mevcut drone hızı (km/h) |
| yaw° | Hedef yönüne açı hatası |
| pit° | Pitch açısı (negatif = ileri eğim) |
| thr | Throttle komutu |
| Durum | OK / YAKIN / GERİ |

---

## 8. Veri Akışı

```
EKF.get_estimated_velocity()
        │ hedef yön vektörü (û)
        ▼
follow_point = target_pos − û × follow_distance_cm
        │
        ▼
s = (drone_pos − follow_point) · û     (imzalı mesafe)
        │
        ▼
dist_pid.calculate(−s)  →  hız düzeltmesi
        │
        ▼
target_speed = 1500 + düzeltme  →  clamp(min, max)
        │
        ├──► speed_pid → pitch_cmd (ileri hız)
        ├──► alt_pid   → throttle  (irtifa)  — eş zamanlı
        └──► yaw_pid   → yaw_cmd   (yön → HEDEFe, follow_point'e değil)
```
