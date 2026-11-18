# 🔥 Firefly — Drone of War Kamikaze AI

**Teknofest "Drone of War" yarışması** için geliştirilmiş otonom kamikaze/takip drone kontrol sistemi. Unreal Engine tabanlı simülatöre TCP üzerinden bağlanarak, gürültülü ve düşük frekanslı (1 Hz) hedef GPS'inden sensör füzyonu (EKF/UKF) ile temiz hedef durumu kestirir ve güdüm + PID ile hedefi kovalayıp yakalar.

---

> ### 🤖 Yapay Zeka Araçları & Hızlı Bağlam
> Projeyi baştan taramadan anlamak için önce şu 3 otorite dosyasını okuyun:
> | Dosya | İçerik |
> |---|---|
> | [`AI_PROJE_OZETI.md`](AI_PROJE_OZETI.md) | Detaylı proje özeti, mimari, ana algoritmalar |
> | [`AI_DOSYA_YAPISI.md`](AI_DOSYA_YAPISI.md) | Dosya haritası — hangi dosya ne işe yarar (durum etiketleriyle) |
> | [`AI_GOREVLER.md`](AI_GOREVLER.md) | Yapılan / devam eden / yapılacak görevler |
>
> Bu dosyalar kod tabanının güncel gerçeğini yansıtır; kodla çeliştiğinde onları güncelleyin.

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

## 🆕 Güncel Durum (2026-11-18)

- **Yarışma modu iz takibi:** Kontrol yolu sadece şartnamedeki telemetriyi kullanır (hedef:
  enlem/boylam/irtifa/yer hızı ≤5 Hz; avcı: GPS + yer hızı + duruş). →
  [`docs/yarisma_modu_takip.md`](docs/yarisma_modu_takip.md)
- **Simülatör çökmesi çözüldü:** `drone_sdk.py` komutları `\n\0` ile bitirir. Sim'i açmak için
  `scripts/sim_up.sh`. → [`docs/simulator_ve_ortam.md`](docs/simulator_ve_ortam.md)
- **GPU:** RF-DETR proje `.venv`'inde CUDA ile ~25 ms/kare.

```bash
scripts/sim_up.sh
python3 -m src.python.simulation.follow_runner --seconds 100 --follow 4000
```

## 📁 Proje Yapısı (Monorepo)

> Ayrıntılı ve durum etiketli tam harita için **[`AI_DOSYA_YAPISI.md`](AI_DOSYA_YAPISI.md)**.

```
FIREFLY (Monorepo)
├── config/                      # Ortak konfigürasyon dosyaları (.yaml)
│   ├── pid_params.yaml          # Uçuş limitleri, iç/dış PID, pursuit güdüm
│   ├── ekf_params.yaml          # CTRV EKF + legacy filtre parametreleri
│   └── competition_server.yaml  # Yarışma sunucusu bağlantı ayarları
├── docs/                        # Rehberler, mimari şemaları ve yarışma notları
├── analysis/                    # Ölçüm/analiz betikleri (data/ git dışı)
├── object_detection/            # RF-DETR ONNX modeli (git dışı)
├── scripts/                     # Kurulum, sim_up.sh, simülasyon/uçuş başlatma
├── src/
│   ├── python/                  # Python Backend Servisi
│   │   ├── navigation/          # Sensör füzyonu / hedef durum kestirimi
│   │   │                        #   fixed_wing_tracker.py (CTRV UKF — ANA takip)
│   │   │                        #   route_intercept_system.py (elips rota + intercept)
│   │   │                        #   ekf.py (config-driven CTRV EKF), simulate_1hz_test.py
│   │   ├── guidance/            # Güdüm + kontrol
│   │   │                        #   pursuit_guidance.py (v2: Gate+LeadSmoother+YawController)
│   │   │                        #   pid_core/pid_pitch/pid_yaw/pid_throttle, mode_manager.py
│   │   ├── vision/              # RF-DETR tespit iskeleti + rfdetr_inference/ (ONNX/TensorRT/SAHI)
│   │   ├── comms/               # Sunucu iletişimi & telemetri (iskelet)
│   │   └── simulation/          # sim_bridge.py (iskelet)
│   └── gcs/                     # Yer Kontrol İstasyonu (Node.js/Electron + Web)
│       ├── server.js            # TCP↔WebSocket köprüsü + HTTP (WS:8765, HTTP:8080)
│       ├── main.js / preload.js / index.js   # Electron ana süreç + köprü
│       ├── renderer/            # React bileşenleri (MapView, TelemetryPanel, ...)
│       ├── web/                 # Tarayıcı arayüzü (harita, telemetri, kontroller)
│       └── assets/              # İkonlar ve stiller
├── tests/                       # python/ ve gcs/ testleri
├── AI_PROJE_OZETI.md            # Proje özeti (AI otorite)
├── AI_DOSYA_YAPISI.md           # Dosya haritası (AI otorite)
├── AI_GOREVLER.md               # Görev listesi (AI otorite)
├── requirements.txt             # Python backend bağımlılıkları
├── package.json                 # GCS ve Node.js bağımlılıkları
├── drone_gui.py                 # (Eski) Tkinter GUI & veri toplama
├── drone_sdk.py                 # TCP İletişim SDK'sı
└── README.md                    # Bu doküman
```

---

## 📄 Lisans

Bu proje Teknofest "Drone of War" yarışması kapsamında geliştirilmiştir.

