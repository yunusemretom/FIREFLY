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

### 2.3 Hız Düzenlemesi — Dış PID

İmzalı mesafe hatası bir **PID kontrolcüsüne** beslenir:

```
hata = 0 − s = −s
dist_correction = dist_pid.calculate(hata, dt)
target_speed = hedef_uçak_hızı + dist_correction
target_speed = clamp(target_speed, min_speed, max_speed)
```

Örnek sayısal değerler (`kp=0.5`, `kd=0.15`):

| s (cm) | Durum | Hedef hız (cm/s) |
|---|---|---|
| −1000 | 10 m geride | ~2000 (hızlanıyor) |
| −500 | 5 m geride | ~1750 |
| 0 | Mükemmel | 1500 (hedefle eş) |
| +200 | 2 m yakın | ~1400 (yavaşlıyor) |
| +600 | 6 m yakın | ~1200 |
| +1500 | 15 m yakın | ~750 (önemli frenleme) |

**D terimi** yaklaşma/uzaklaşma *hızına* tepki verir: mesafe kapanırken önceden frenlemeye başlar → overshoot azalır, daha yumuşak geçiş olur.

**I terimi** küçük sabit kaymaları ortadan kaldırır (örn. rüzgar vb. sürekli disturbance).

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
                                # 500  →  5 m
                                # 1000 → 10 m (varsayılan)
                                # 2000 → 20 m

  distance_pid:                 # Dış döngü: mesafe → hız
    kp: 0.5                     # Orantısal: 100 cm hata → 50 cm/s düzeltme
    ki: 0.02                    # İntegral: sabit kayma giderir
    kd: 0.15                    # Türev: yaklaşma hızına önceden tepki
    int_limit: 200.0            # İntegral sınırı ±200 cm/s

  min_speed_cms: 200.0          # Minimum ileri hız (durma engeli)
  max_speed_cms: 2800.0         # Maksimum yaklaşma hızı (~100 km/h)
```

### Parametre Etkileri

| Parametre | Küçük değer | Büyük değer |
|---|---|---|
| `follow_distance_cm` | Daha yakın takip, çarpışma riski artar | Daha güvenli ama daha az agresif |
| `distance_pid.kp` | Yavaş tepki, mesafe stabilitesi zayıf | Hızlı tepki ama sallantı/overshoot riski |
| `distance_pid.kd` | Overshoot artar | Frenleme erken başlar, daha yumuşak |
| `distance_pid.ki` | Sabit kayma kalabilir | Kayma giderilir ama windup riski |
| `min_speed_cms` | Drone durabilir/geri gidebilir | Drone her zaman biraz ilerler |
| `max_speed_cms` | Yaklaşma yavaşlar | Daha hızlı kapanma |

### Hızlı Ayarlama Kılavuzu

**Drone çok yaklaşıp geri çekiliyor (salınım):**
→ `kp` düşür veya `kd` artır

**Drone hedefi takip edemeyip geri kalıyor:**
→ `kp` artır veya `max_speed_cms` artır

**Drone yavaş yavaş yaklaşıp uzaklaşıyor (drift):**
→ `ki` küçük artır

**Mesafe doğruyken hız çok fazla/az:**
→ İç hız döngüsü gainlerini ayarla (`outer_pid.speed` → `pid_params.yaml`)

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
