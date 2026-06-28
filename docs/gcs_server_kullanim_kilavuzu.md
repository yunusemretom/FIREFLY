# 🔥 FIREFLY GCS — Web Sunucusu Kullanım Kılavuzu

> **Versiyon:** 1.0 · **Güncelleme:** 2026-06-28  
> **Dosya:** `src/gcs/server.js`

---

## İçindekiler

1. [Genel Bakış](#1-genel-bakış)
2. [Gereksinimler](#2-gereksinimler)
3. [İlk Kurulum](#3-ilk-kurulum)
4. [Sunucuyu Başlatma](#4-sunucuyu-başlatma)
5. [Web Arayüzüne Bağlanma](#5-web-arayüzüne-bağlanma)
6. [Drone Simülatörüne Bağlanma](#6-drone-simülatörüne-bağlanma)
7. [Yapılandırma (Port Değiştirme)](#7-yapılandırma-port-değiştirme)
8. [Sunucuyu Durdurma](#8-sunucuyu-durdurma)
9. [Mimari — Ne Nereye Bağlanır?](#9-mimari--ne-nereye-bağlanır)
10. [Sorun Giderme](#10-sorun-giderme)
11. [Protokol Referansı](#11-protokol-referansı)

---

## 1. Genel Bakış

FIREFLY GCS sunucusu, Unreal Engine tabanlı drone simülatörü ile web tarayıcısı arasında köprü görevi gören bir **Node.js** uygulamasıdır.

Tek bir komutla üç farklı sunucu birden başlatılır:

| Servis | Port | Açıklama |
|--------|------|----------|
| **HTTP** | `8080` | Web arayüzünü tarayıcıya sunar (`index.html`) |
| **WebSocket** | `8765` | Tarayıcı ↔ Sunucu gerçek zamanlı iletişimi |
| **TCP (Drone)** | `12345` | Unreal Engine simülatörüne bağlantı (siz başlatırsınız) |

```
Tarayıcı
    │  ws://localhost:8765
    ▼
server.js (Node.js)
    │  TCP :12345
    ▼
Unreal Engine Simülatörü
```

---

## 2. Gereksinimler

### Zorunlu

| Gereksinim | Minimum Versiyon | Nasıl Kontrol Edilir |
|-----------|-----------------|----------------------|
| **Node.js** | v12 veya üzeri | `node --version` |
| **npm** | v6 veya üzeri | `npm --version` |

### İsteğe Bağlı

| Gereksinim | Ne İçin |
|-----------|---------|
| **Unreal Engine Simülatörü** | Gerçek telemetri verisi almak için |
| Modern tarayıcı (Chrome, Firefox, Edge) | Web arayüzü için |

### Node.js Yüklü Değilse

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install nodejs npm
```

**Linux (NodeSource — güncel versiyon için):**
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs
```

**Versiyon doğrulama:**
```bash
node --version   # v12.x veya üzeri çıkmalı
npm --version    # v6.x veya üzeri çıkmalı
```

---

## 3. İlk Kurulum

Sunucuyu **ilk kez** çalıştırmadan önce bağımlılıkların yüklenmesi gerekir. Bu işlem **tek seferlik** yapılır.

### Yöntem A — GCS klasöründen (önerilen)

```bash
cd /path/to/Firefly/src/gcs
npm install
```

### Yöntem B — Proje kök dizininden

```bash
cd /path/to/Firefly
npm install --prefix src/gcs
```

Kurulum başarılıysa `src/gcs/node_modules/` klasörü oluşur ve şu paketler yüklenir:

- `express` — HTTP sunucu ve statik dosya servisi
- `ws` — WebSocket sunucu

> **Not:** `node_modules/` klasörü Git'e eklenmez (`.gitignore` tarafından hariç tutulur).  
> Projeyi klonladıktan veya güncelledikten sonra `npm install` komutunu tekrar çalıştırın.

---

## 4. Sunucuyu Başlatma

Bağımlılıklar yüklendikten sonra sunucu şu yöntemlerden biriyle başlatılabilir:

### Yöntem 1 — Proje kök dizininden (en kolay)

```bash
cd /path/to/Firefly
npm run gcs
```

### Yöntem 2 — GCS klasöründen

```bash
cd /path/to/Firefly/src/gcs
node server.js
```

### Yöntem 3 — `npm start` (GCS klasöründen)

```bash
cd /path/to/Firefly/src/gcs
npm start
```

### Başarılı Başlatma Çıktısı

Sunucu başarıyla başlatıldığında terminalde şu çıktı görünür:

```
[GCS INFO] ============================================================
[GCS INFO]   🔥 FIREFLY GCS — Node.js  (HTTP + WS + TCP Drone Köprüsü)
[GCS INFO] ============================================================
[GCS INFO]   Web Arayüzü: http://localhost:8080
[GCS INFO]   WebSocket:   ws://localhost:8765
[GCS INFO]   Drone TCP:   127.0.0.1:12345  (ayarlar panelinden değiştir)
[GCS INFO]   Timeout:     TCP bağlantı 5s | İnaktivite 5s
[GCS INFO] ============================================================
[GCS INFO] HTTP sunucusu: http://localhost:8080
[GCS INFO] WebSocket dinleniyor: ws://localhost:8765
```

Bu çıktıyı gördükten sonra sunucu hazırdır.

---

## 5. Web Arayüzüne Bağlanma

Sunucu çalışırken herhangi bir tarayıcıda şu adresi açın:

```
http://localhost:8080
```

Arayüz açıldığında:

- **Sağ üst köşe** → "DRONE YOK" (kırmızı nokta) — simülatöre henüz bağlanılmadı
- **⚙ AYARLAR sekmesi** → "GCS SUNUCU — Bağlı" (yeşil nokta) — web sunucusu çalışıyor

### Sekmeler

| Sekme | İçerik |
|-------|--------|
| **COCKPIT** | Kamera görüntüsü, telemetri verileri, kontrol sliderları |
| **MAPS** | Drone ve hedefin interaktif harita üzerinde konumu |
| **AYARLAR** | Bağlantı ayarları, PID parametreleri, kontrol limitleri |
| **HAKKIMIZDA** | Sistem mimarisi ve versiyon bilgisi |

---

## 6. Drone Simülatörüne Bağlanma

Web arayüzü açıldıktan sonra drone simülatörüne bağlanmak için:

1. **Önce** Unreal Engine simülatörünü başlatın (TCP port `12345`'i dinlemeli)
2. Tarayıcıda **⚙ AYARLAR** sekmesine gidin
3. **🔗 Drone Bağlantısı** kartında:
   - **Simülatör Host:** `127.0.0.1` (yerel makine) veya simülatörün IP adresi
   - **Port:** `12345` (varsayılan)
   - **Timeout:** `5` saniye
4. Sayfanın herhangi bir yerindeki **BAĞLAN** butonuna tıklayın

### Bağlantı Durumları

| Durum | Topbar Gösterge | Açıklama |
|-------|----------------|----------|
| Bağlanıyor | 🟡 yanıp sönüyor | TCP bağlantı denemesi devam ediyor |
| Bağlandı | 🟢 DRONE BAĞLI | Telemetri akıyor |
| Bağlantı yok | 🔴 DRONE YOK | Simülatör kapalı veya yanlış IP/port |
| Timeout | 🔴 DRONE YOK | 5 saniyede yanıt gelmedi |

### 5 Saniyelik Timeout Kuralı

- **Bağlantı Timeout:** Simülatör 5 saniye içinde TCP bağlantısını kabul etmezse otomatik iptal edilir ve uyarı gösterilir.
- **İnaktivite Timeout:** Bağlantı kurulduktan sonra 5 saniye boyunca veri gelmezse bağlantı kesilir.

---

## 7. Yapılandırma (Port Değiştirme)

Varsayılan portlar zaten kullanımdaysa ortam değişkenleriyle değiştirilebilir:

```bash
# HTTP portunu 9090, WebSocket portunu 9765 olarak değiştir
GCS_HTTP_PORT=9090 GCS_WS_PORT=9765 node src/gcs/server.js
```

### Mevcut Yapılandırma Seçenekleri

| Ortam Değişkeni | Varsayılan | Açıklama |
|----------------|-----------|----------|
| `GCS_HTTP_PORT` | `8080` | Web arayüzü HTTP portu |
| `GCS_WS_PORT` | `8765` | WebSocket sunucu portu |

> **Drone TCP portu** (`12345`) ortam değişkeniyle değil, web arayüzündeki  
> **⚙ AYARLAR → Drone Bağlantısı** kartından değiştirilir.

---

## 8. Sunucuyu Durdurma

Terminalde sunucunun çalıştığı pencerede:

```
Ctrl + C
```

Sunucu düzgün kapanır ve şu mesajı gösterir:

```
[GCS INFO] GCS sunucusu kapatılıyor… (SIGINT)
```

---

## 9. Mimari — Ne Nereye Bağlanır?

```
┌─────────────────────────────────────────────────────────┐
│                    FIREFLY GCS                          │
│                                                         │
│   Tarayıcı (Chrome/Firefox)                             │
│   ┌────────────────────────┐                            │
│   │  http://localhost:8080 │◄──── HTTP (statik dosyalar)│
│   │  ws://localhost:8765   │◄──── WebSocket (20 Hz)     │
│   └────────────────────────┘                            │
│              ▲ ▼                                        │
│   ┌──────────────────────────────┐                      │
│   │   server.js (Node.js)        │                      │
│   │   ├── express (HTTP :8080)   │                      │
│   │   ├── ws (WebSocket :8765)   │                      │
│   │   └── net.Socket (TCP)       │                      │
│   └──────────────────────────────┘                      │
│              ▲ ▼                                        │
│   ┌────────────────────────┐                            │
│   │  Unreal Engine         │                            │
│   │  Simülatörü (TCP:12345)│                            │
│   └────────────────────────┘                            │
└─────────────────────────────────────────────────────────┘
```

### Veri Akışı

**Telemetri (Simülatör → Tarayıcı):**
```
Unreal Engine
  → TCP CSV satırı (18 float, "\n" sonlu)
  → server.js parse eder
  → WebSocket JSON olarak tarayıcıya iletilir (20 Hz)
  → Tarayıcıda göstergeler güncellenir
```

**Kontrol Komutu (Tarayıcı → Simülatör):**
```
Tarayıcıda slider hareketi
  → WebSocket JSON: {"type":"control", "throttle":0.5, ...}
  → server.js alır
  → TCP "throttle,pitch,roll,yaw,arm\n" olarak simülatöre gönderilir
```

---

## 10. Sorun Giderme

### Hata: `Error: Cannot find module 'express'`

**Sebep:** `npm install` çalıştırılmamış.

**Çözüm:**
```bash
cd src/gcs
npm install
```

---

### Hata: `Error: listen EADDRINUSE :::8080`

**Sebep:** Port 8080 başka bir uygulama tarafından kullanılıyor.

**Çözüm A — Portu değiştir:**
```bash
GCS_HTTP_PORT=9080 npm run gcs
```

**Çözüm B — Portu kullanan uygulamayı bul ve kapat:**
```bash
sudo lsof -i :8080
sudo kill <PID>
```

---

### Hata: `connect ECONNREFUSED 127.0.0.1:12345`

**Sebep:** Unreal Engine simülatörü çalışmıyor veya farklı bir portta.

**Çözüm:**
1. Simülatörü başlatın
2. Web arayüzünden doğru IP ve portu girin
3. **BAĞLAN** butonuna tekrar tıklayın

---

### Tarayıcıda sayfa açılmıyor

**Kontrol listesi:**
```bash
# Sunucunun çalıştığını doğrula
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080
# → 200 çıkmalı

# Portu dinleyen process'i kontrol et
ss -tlnp | grep 8080
```

---

### `npm run gcs` komutu bulunamıyor

Projenin **kök dizininde** (`Firefly/`) olduğunuzu kontrol edin:

```bash
pwd
# /home/kullanici/Projeler/Firefly  ← olmalı

npm run gcs
```

---

### Bağlantı kuruluyor ama telemetri gelmiyor

- Simülatörün TCP port üzerinden veri gönderdiğini doğrulayın
- 5 saniye içinde veri gelmezse sunucu otomatik bağlantıyı keser (inaktivite timeout)
- Simülatörü yeniden başlatıp **BAĞLAN** butonuna tekrar tıklayın

---

## 11. Protokol Referansı

`drone_sdk.py` ile birebir aynı protokol Node.js'te uygulanmıştır.

### Simülatörden Gelen Veri (TCP → Sunucu)

Her satır `\n` ile biter. 18 virgülle ayrılmış float değeri:

```
v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15],v[16],v[17]\n
```

| İndeks | Alan | Birim |
|--------|------|-------|
| 0–2 | Drone Pozisyon (X, Y, Z) | cm |
| 3–5 | Drone Rotasyon (Roll, Pitch, Yaw) | derece |
| 6–8 | Drone Hız Vektörü (Vx, Vy, Vz) | cm/s |
| 9 | Drone Toplam Hız | cm/s |
| 10 | Drone İrtifa (Z ekseni) | cm |
| 11–13 | Hedef Pozisyon (X, Y, Z) | cm |
| 14–16 | Hedef Rotasyon (Roll, Pitch, Yaw) | derece |
| 17 | Hedef Toplam Hız | cm/s |

### Sunucudan Giden Kontrol Komutu (Sunucu → TCP)

```
throttle,pitch,roll,yaw,arm\n
```

| Alan | Aralık | Açıklama |
|------|--------|----------|
| `throttle` | `-1.0` – `1.0` | `-1` alçal, `0` hover, `+1` tırman |
| `pitch` | `-1.0` – `1.0` | İleri/geri eğilme |
| `roll` | `-1.0` – `1.0` | Sağa/sola yatış |
| `yaw` | `-1.0` – `1.0` | Kendi ekseni etrafında dönme |
| `arm` | `0` veya `1` | `1` = aktif, `0` = pasif |

**Örnek:**
```
0.5000,0.2000,-0.1000,0.0500,1
```

---

*Bu belge FIREFLY GCS projesinin bir parçasıdır.*  
*Kaynak: `src/gcs/server.js` · WebSocket: `src/gcs/web/js/app.js`*
