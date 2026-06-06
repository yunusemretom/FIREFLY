# 🔥 Firefly — Drone of War Kamikaze AI

**Teknofest "Drone of War" yarışması** için geliştirilmiş otonom kamikaze drone kontrol sistemi. Unreal Engine tabanlı simülatöre TCP üzerinden bağlanarak, PID kontrol algoritması ile hedefe otomatik kilitlenme ve saldırı gerçekleştirir.

---

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [Mimari](#mimari)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Kontrol Modları](#kontrol-modları)
- [SDK API Referansı](#sdk-api-referansı)
- [Proje Yapısı](#proje-yapısı)

---

## 🎯 Genel Bakış

Firefly, hareketli bir hava hedefini otonom olarak takip edip vurmak için tasarlanmış bir yapay zeka kontrol sistemidir. Sistem iki ana bileşenden oluşur:

- **Drone SDK** — Unreal Engine simülatörü ile TCP soket iletişimi
- **Drone GUI** — Tkinter tabanlı yer kontrol istasyonu ve otonom mod yönetimi

### Temel Özellikler

| Özellik | Açıklama |
|---|---|
| 🎮 Manuel Kontrol | Throttle, Pitch, Roll, Yaw slider kontrolleri |
| 🤖 Kamikaze Modu | PID tabanlı otonom hedef takip ve saldırı |
| 📡 Canlı Telemetri | Konum, rotasyon, hız, irtifa bilgileri |
| 🔗 TCP İletişim | `localhost:12345` üzerinden Unreal Engine bağlantısı |
| ⚡ 50ms Kontrol Döngüsü | 20Hz güncelleme frekansı |

---

## 🏗️ Mimari

```
┌─────────────────────┐     TCP (12345)     ┌──────────────────────┐
│   Drone GUI (Python) │◄──────────────────►│ Unreal Engine Sim    │
│                       │                    │ (DronesOfWar.exe)    │
│  ┌─────────────────┐ │  Kontrol Komutları  │                      │
│  │ Manuel / Otonom  │─┼──────────────────►│  Drone Fizik Motoru  │
│  │ Kontrol Mantığı  │ │                    │                      │
│  └─────────────────┘ │  Telemetri Verisi   │                      │
│  ┌─────────────────┐ │◄──────────────────│  Hedef Sistemi       │
│  │ Drone SDK       │ │                    │                      │
│  │ (TCP Client)     │ │                    │                      │
│  └─────────────────┘ │                    └──────────────────────┘
└─────────────────────┘
```

---

## 🚀 Kurulum

### Gereksinimler

- **Python 3.8+**
- **Tkinter** (Python ile birlikte gelir)
- Ek bir paket kurulumu gerekmez — proje tamamen standart kütüphaneler kullanır.

### Adımlar

```bash
# Repoyu klonla
git clone https://github.com/<kullanici>/Firefly.git
cd Firefly

# Simülatörü başlat
cd "Drones of War Teknofest"
./DronesOfWar.exe

# GUI'yi çalıştır (yeni terminal)
cd ..
python drone_gui.py
```

---

## 🎮 Kullanım

1. **Simülatörü başlatın** — `Drones of War Teknofest/DronesOfWar.exe`
2. **GUI'yi çalıştırın** — `python drone_gui.py`
3. **"INITIALIZE LINK"** butonuna tıklayarak TCP bağlantısı kurun
4. **"ARM WEAPON SYSTEM"** ile dronu aktif edin
5. Manuel kontrol veya **"ENABLE KAMIKAZE MODE"** ile otonom mod seçin

---

## 🎯 Kontrol Modları

### Manuel Mod

Slider'lar aracılığıyla doğrudan kontrol:

- **THR (Throttle):** 0.0 → 1.0 — Gaz / Yükselme
- **Pitch:** -1.0 → 1.0 — İleri / Geri eğilme
- **Roll:** -1.0 → 1.0 — Sağa / Sola yatış
- **Yaw:** -1.0 → 1.0 — Eksen dönüşü
- **Max Throttle:** Gaz üst limitini sınırlandırma

### Kamikaze Modu (Otonom)

İki fazlı akıllı saldırı algoritması:

| Faz | Mesafe | Strateji |
|---|---|---|
| **Faz 1 — Yakalama** | > 45m | Agresif dalış açısı, maksimum hız |
| **Faz 2 — Hassas Yaklaşım** | ≤ 45m | Kademeli burun düzeltme, flare manevrası |

**PID Kazançları:**
- `KP_YAW = 0.07` — Yaw yönelme hassasiyeti
- `KP_PITCH = 0.035` — Pitch kontrol hassasiyeti
- `KP_THR = 0.01` — Throttle düzeltme hassasiyeti

---

## 📡 SDK API Referansı

### Bağlantı

```python
import drone_sdk as drone

drone.connect(host='127.0.0.1', port=12345)  # Simülatöre bağlan
drone.is_connected()                          # Bağlantı durumu
drone.disconnect()                            # Bağlantıyı kapat
```

### Kontrol Fonksiyonları

```python
drone.set_throttle(0.5)     # Gaz [0.0 - 1.0]
drone.set_pitch(0.3)        # İleri/Geri [-1.0 - 1.0]
drone.set_roll(-0.2)        # Sağa/Sola [-1.0 - 1.0]
drone.set_yaw(0.1)          # Eksen dönüşü [-1.0 - 1.0]
drone.set_arm(True)         # Dronu aktif et

# Tek paket gönderimi (önerilen)
drone.set_control_surfaces(throttle, pitch, roll, yaw, arm)
```

### Telemetri Fonksiyonları

```python
drone.get_drone_location()   # (X, Y, Z) konum
drone.get_drone_rotation()   # (Roll, Pitch, Yaw) rotasyon
drone.get_drone_speed()      # cm/s hız
drone.get_drone_altitude()   # cm irtifa
drone.get_target_location()  # (X, Y, Z) hedef konumu
drone.get_target_speed()     # cm/s hedef hızı
drone.get_telemetry()        # Tüm veri (dict)
```

### TCP Protokolü

```
Gönderme: throttle,pitch,roll,yaw,arm\n
Alma:     droneX,droneY,droneZ,roll,pitch,yaw,vx,vy,vz,speed,alt,tarX,tarY,tarZ,tarRoll,tarPitch,tarYaw,tarSpeed\n
```

---

## 📁 Proje Yapısı

```
Firefly/
├── drone_gui.py                  # Tkinter GUI + Kamikaze AI mantığı
├── drone_sdk.py                  # TCP iletişim SDK'sı (v2.1)
├── Drones of War Teknofest/      # Unreal Engine simülatör dosyaları
│   ├── DronesOfWar.exe           # Oyun çalıştırılabilir dosyası
│   ├── DronesOfWar/              # Oyun veri dosyaları
│   ├── Engine/                   # Unreal Engine çalışma dosyaları
│   └── SDL2.dll                  # SDL2 kütüphanesi
├── .gitignore
└── README.md
```

---

## 📄 Lisans

Bu proje Teknofest "Drone of War" yarışması kapsamında geliştirilmiştir.
