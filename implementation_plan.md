# FIREFLY Monorepo Klasör Yapısı Kurulum Planı

Bu çalışma planı, **FIREFLY** otonom FPV drone yakalama/müdahale sistemi için kapsamlı bir monorepo klasör yapısının kurulmasını hedefler. Proje; Python backend servisleri, Node.js/Electron tabanlı Yer Kontrol İstasyonu (GCS), ortak konfigürasyonlar, testler, dokümanlar ve uçuş/simülasyon başlatma betiklerini içerecektir.

## Kullanıcı İncelemesi Gereken Konular

> [!IMPORTANT]
> **Kök Dizin Seçimi:** Çalışma alanı dizininiz halihazırda `/home/tom/Documents/Projeler/Firefly` olarak belirlenmiştir. Monorepo yapısını doğrudan bu dizin altında (yani bu dizini `FIREFLY` olarak kabul ederek) oluşturmayı planlıyoruz. Eğer yeni bir `FIREFLY` alt klasörü oluşturup yapıyı onun içine kurmamızı tercih ederseniz lütfen belirtiniz.

> [!NOTE]
> **Mevcut Dosyalar:** Mevcut `drone_gui.py` ve `drone_sdk.py` dosyaları korunacaktır. `README.md` dosyası yeni monorepo yapısını ve mimariyi yansıtacak şekilde güncellenecektir.

## Önerilen Değişiklikler

Monorepo yapısında oluşturulacak klasörler ve yer tutucu (placeholder) dosyalar aşağıda listelenmiştir:

### 1. Python Backend Servisi (`src/python/`)
Python tarafındaki her alt modül için uygun `__init__.py` dosyaları ve temel kod iskeletlerini barındıran yer tutucu Python dosyaları oluşturulacaktır.

- **`src/python/navigation/` (Navigasyon):** Bozuk GNSS verileri için EKF sensör füzyonu.
  - `ekf.py`
  - `waypoint_controller.py`
  - `gnss_filter.py`
  - `__init__.py`
- **`src/python/vision/` (Görüntü İşleme):** RT-DETR nesne tespiti ve TCT-Track izleme.
  - `detector.py`
  - `tracker.py`
  - `lock_detector.py`
  - `__init__.py`
- **`src/python/guidance/` (Rehberlik/Yönlendirme):** PID kontrolcüleri ve CRSF paket üretimi.
  - `pid_yaw.py`
  - `pid_pitch.py`
  - `pid_throttle.py`
  - `crsf_sender.py`
  - `mode_manager.py`
  - `__init__.py`
- **`src/python/comms/` (Haberleşme):** Yarışma sunucusu iletişimi ve telemetri.
  - `server_client.py`
  - `telemetry_parser.py`
  - `data_broker.py`
  - `__init__.py`
- **`src/python/simulation/` (Simülasyon):** Gazebo/ROS2 entegrasyonu ve GNSS gürültü üreteci.
  - `gnss_jammer.py`
  - `sim_bridge.py`
  - `scenario_runner.py`
  - `__init__.py`

### 2. Node.js/Electron Yer Kontrol İstasyonu (`src/gcs/`)
Electron ana süreci, React bileşenleri, socket.io sunucusu ve stil/ikon dosyaları için yer tutucu dosyalar oluşturulacaktır.

- **Main Process (Ana Süreç):**
  - `src/gcs/main.js` (Electron ana giriş noktası)
  - `src/gcs/preload.js`
- **Renderer (React UI Bileşenleri):**
  - `src/gcs/renderer/MapView.jsx`
  - `src/gcs/renderer/VideoOverlay.jsx`
  - `src/gcs/renderer/LockIndicator.jsx`
  - `src/gcs/renderer/TelemetryPanel.jsx`
  - `src/gcs/renderer/StatusBar.jsx`
  - `src/gcs/renderer/ServerStatus.jsx`
  - `src/gcs/renderer/index.js` (Yer tutucu)
- **Backend Bridge (Socket.io):**
  - `src/gcs/index.js` (Python servislerine bağlanan socket.io sunucusu)
- **Assets (Varlıklar):**
  - `src/gcs/assets/styles.css`
  - `src/gcs/assets/icons/README.md` (Yer tutucu)

### 3. Paylaşılan Yapılandırma ve Belgeler (Kök Seviye)
- **`config/` (Yapılandırma Dosyaları):**
  - `pid_params.yaml`
  - `ekf_params.yaml`
  - `competition_server.yaml`
  - `crsf_config.yaml`
- **`tests/` (Test Aynalaması):**
  - `tests/python/` (python modülleri için birim test yer tutucuları)
  - `tests/gcs/` (gcs için birim test yer tutucuları)
- **`docs/` (Dokümantasyon):**
  - `architecture_diagrams.md`
  - `competition_notes.md`
  - `README.md` (Yer tutucu)
- **`scripts/` (Kurulum ve Başlatma Betikleri):**
  - `setup.sh` (Kurulum betiği)
  - `launch_simulation.sh` (Simülasyon başlatma betiği)
  - `launch_real_flight.sh` (Gerçek uçuş başlatma betiği)
- **Kök Klasör Dosyaları:**
  - `.env.example`
  - `requirements.txt`
  - `package.json`
  - `README.md` (Yeni monorepo yapısını açıklayan güncellenmiş dosya)

## Doğrulama Planı

Klasör yapısının doğruluğunu teyit etmek için aşağıdaki adımlar uygulanacaktır:
1. `tree` komutu veya alternatif bir listeleme komutu ile tüm yapının doğru yerleşimde oluşturulduğu doğrulanacaktır.
2. Oluşturulan dosyaların içeriklerinin sintaks hatalarından arındırılmış olduğu (örneğin boş `__init__.py`'ler, geçerli YAML şablonları, geçerli JSON formatı) kontrol edilecektir.
3. Node.js için `package.json` ve Python için `requirements.txt` dosyalarının temel bağımlılık tanımları doğrulanacaktır.
