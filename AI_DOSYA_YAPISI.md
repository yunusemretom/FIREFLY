<!--
================================================================================
  YAPAY ZEKA ARAÇLARI İÇİN NOT (AI ASSISTANT DIRECTIVE)
  Bu dosya, Firefly kod tabanının OTORİTE dosya haritasıdır. Kod tabanını
  baştan taramak yerine burayı oku. Kodu değiştirdiğinde bu haritayı güncelle.
  Kardeş dosyalar: AI_PROJE_OZETI.md, AI_GOREVLER.md   |   Güncel: 2026-07-07
================================================================================
-->

# 📁 Firefly — Dosya Yapısı Haritası

> **Durum etiketleri:** ⭐ aktif/ana yol · 🧪 test/analiz · 🗂️ iskelet/placeholder ·
> ⚠️ eski/eskimiş · 🔧 araç

```
Firefly/
├── AI_PROJE_OZETI.md          ⭐ Proje özeti (AI için otorite)
├── AI_DOSYA_YAPISI.md         ⭐ Bu dosya
├── AI_GOREVLER.md             ⭐ Görev listesi (yapılan/yapılacak)
│
├── drone_sdk.py               ⭐ TCP SDK — simülatör iletişimi (parse+kontrol)
├── drone_gui.py               ⚠️ Eski Tkinter kontrol paneli + Excel veri toplama
├── README.md                  ⚠️ Kısmen eski (silinmiş modüllere atıf var)
├── implementation_plan.md     📄 İlk monorepo kurulum planı (tarihsel)
├── Promtlar.md                📄 EKF/pursuit için kullanılan üretim prompt'ları
├── requirements.txt           📦 Python bağımlılıkları
├── package.json               📦 Kök Node bağımlılıkları
├── hedef_konumlari.xlsx        📊 Kalibrasyon/kayıt verisi (gerçek+gürültülü GPS)
├── ekf_loiter_sim.png          📊 EKF loiter simülasyon çıktısı
├── pursuit_smoothing_karsilastirma.png  📊 Pursuit v1↔v2 A/B grafiği
│
├── config/                                # Ayar dosyaları (kod hard-code etmez)
│   ├── pid_params.yaml        ⭐ Uçuş limitleri, iç/dış PID, pursuit güdüm
│   ├── ekf_params.yaml        ⭐ ctrv_ekf + legacy target_ekf parametreleri
│   └── competition_server.yaml   Yarışma sunucusu host/port/team_id
│
├── src/python/                            # PYTHON BEYİN
│   ├── navigation/                        # Sensör füzyonu / hedef durum kestirimi
│   │   ├── fixed_wing_tracker.py   ⭐ CTRV UKF tracker (7 durum) — ANA takip
│   │   ├── route_intercept_system.py ⭐ Elips rota fit + RoutePlanner + intercept
│   │   ├── orbit_lock.py           ⭐ OrbitLockTracker: tur tespiti + kilitle/dondur
│   │   ├── telemetry_conditioner.py ⭐ MeasurementGate + AltitudeLock (stale-hold koşullandırma)
│   │   ├── ekf.py                  ⭐ Config-driven CTRV EKF (5 durum, analitik Jacobian)
│   │   ├── ekf_deneme.py           🧪 EKF deneme/geliştirme sürümü
│   │   └── simulate_1hz_test.py    🧪 1 Hz koşulu: UKF vs CV Kalman kıyası
│   │
│   ├── guidance/                          # Güdüm + kontrol
│   │   ├── pursuit_guidance.py     ⭐ v2: Gate + LeadSmoother + YawController
│   │   ├── route_follow_guidance.py ⭐ Elips rota takip güdümü (carrot + hız kanalı) + GuidePointSlewLimiter
│   │   ├── mode_manager.py         🗂️ Uçuş modu geçiş mantığı (iskelet)
│   │   ├── pid_core.py             ⭐ Genel PID çekirdeği
│   │   ├── pid_pitch.py            ⭐ Pitch kontrolcüsü
│   │   ├── pid_yaw.py              🗂️ Yaw PID
│   │   └── pid_throttle.py         🗂️ Throttle PID
│   │
│   ├── vision/                            # Hedef tespiti
│   │   ├── detector.py             🗂️ RT-DETR dedektör iskeleti
│   │   ├── tracker.py              🗂️ Görsel takip iskeleti
│   │   ├── lock_detector.py        🗂️ Kilitlenme tespiti iskeleti
│   │   └── rfdetr_inference/       🔧 RF-DETR çalıştırma/export araçları
│   │       ├── export_rfdetr_onnx.py
│   │       ├── export_rfdetr_tensorrt.py
│   │       ├── run_rfdetr_tensorrt.py
│   │       ├── run_rfdetr_video_test.py
│   │       ├── run_rfdetr_screen_capture.py
│   │       └── run_rfdetr_sahi_test.py
│   │   └── sort_csrt_tracking/       🔧 SORT+CSRT çoklu nesne tracker (klonlanmış repo)
│   │       └── sort_csrt.py           ⭐ Sort + CSRTTracker sınıfları
│   │
│   ├── comms/                      🗂️ Sunucu iletişimi/telemetri parser (iskelet)
│   │   ├── server_client.py
│   │   ├── telemetry_parser.py
│   │   └── data_broker.py
│   │
│   └── simulation/
│       └── sim_bridge.py          🗂️ Simülasyon köprüsü (iskelet)
│
├── src/gcs/                               # YER KONTROL İSTASYONU (Node.js)
│   ├── server.js              ⭐ TCP↔WebSocket köprü + HTTP (WS:8765, HTTP:8080)
│   ├── main.js / preload.js / index.js   Electron ana süreç + köprü
│   ├── renderer/              🗂️ React bileşenleri (MapView, VideoOverlay,
│   │                             LockIndicator, TelemetryPanel, StatusBar, ServerStatus)
│   ├── web/                   ⭐ Tarayıcı arayüzü (index.html + js/{app,map,controls,instruments})
│   ├── assets/                   styles.css, ikonlar
│   └── package.json / package-lock.json
│
├── tests/
│   ├── python/
│   │   ├── test_fixed_wing_tracker.py  🧪 UKF tracker testi
│   │   ├── test_ekf.py                 🧪 EKF testi (loiter sim + matplotlib)
│   │   ├── test_route_intercept.py     🧪 Rota takip testi (İNCE harness; modüllerden çeker)
│   │   ├── test_route_kamikaze.py      🧪 Rota kamikaze taarruz testi (faz-demirli intercept)
│   │   ├── test_rfdetr_sahi_csrt.py    🧪 RF-DETR SAHI + SORT-CSRT video tracking testi
│   │   ├── test_rfdetr_trt_visual_servo.py 🧪 TRT+SAHI+CSRT ekran testi: merkez kilitlenme dörtgeni + G tuşuyla görsel servolama (drone_sdk)
│   │   ├── test_rfdetr_tctrack.py      🧪 RF-DETR + TCTrack otomatik faz dongusu testi
│   │   ├── route_viz.py                🔧 Ortak test görselleştirme (tema+pano+elips sim hedefi)
│   │   ├── test_pursuit_smoothing.py   🧪 Canlı-koşul emülasyon harness (A/B/C)
│   │   ├── test_pursuit_live.py        ⭐🧪 Düzeltilmiş CANLI pursuit runner
│   │   └── pursuit_guidance (1).py     ⚠️ v1 (dalgalı yol) — referans/kıyas
│   ├── gcs/test_bridge.js              🧪 GCS köprü testi
│   └── README.md
│
├── docs/                                  # Rehberler (bazıları silinmiş modüllere atıf yapar ⚠️)
│   ├── README.md                   Doküman indeksi
│   ├── ekf_rehberi.md              EKF filtresi rehberi
│   ├── pursuit_rehberi.md          Pursuit güdüm rehberi
│   ├── sim_bridge_ekf_rehberi.md   ⚠️ Silinmiş bridge'lere atıf
│   ├── pid_ve_sim_bridge_rehberi.md
│   ├── gcs_server_kullanim_kilavuzu.md  ⭐ GCS web sunucu kullanımı
│   ├── architecture_diagrams.md
│   └── competition_notes.md
│
└── scripts/
    ├── setup.sh                🔧 Kurulum
    ├── launch_simulation.sh    🔧 Simülasyon başlat
    └── launch_real_flight.sh   🔧 Gerçek uçuş başlat
```

---

## Kritik Dosyalar — Nereye Bakmalı?

| İhtiyaç | Dosya |
|---|---|
| Simülatörle konuşmak | `drone_sdk.py` |
| Hedefi filtrelemek (ANA) | `src/python/navigation/fixed_wing_tracker.py` |
| Pürüzsüz kovalama güdümü | `src/python/guidance/pursuit_guidance.py` |
| Rota öğrenip yakalama noktası | `src/python/navigation/route_intercept_system.py` |
| Config-driven EKF | `src/python/navigation/ekf.py` |
| PID/uçuş ayarları | `config/pid_params.yaml` |
| Filtre ayarları | `config/ekf_params.yaml` |
| Canlı test | `tests/python/test_pursuit_live.py` |
| Web GCS köprüsü | `src/gcs/server.js` |

## Git Bağlamı (2026-07-04 itibarıyla)

- **Silinmiş legacy modüller** (artık YOK, atıf yapan doküman eskimiştir):
  `crsf_sender.py`, `pursuit.py`, `gnss_filter.py`, `nav_integration.py`,
  `waypoint_controller.py`, `gnss_jammer.py`, `scenario_runner.py`,
  `test_ekf_online.py`, `live_ekf_monitor.py`, `sim_bridge_ekf.py`,
  `sim_bridge_pursuit.py`, `crsf_config.yaml`, ayrıca kök `canlı_ekf.py`,
  `ekf_dogruluk.py`, `nisan.py`.
- **Yeni eklenen ana yol:** `pursuit_guidance.py`, `fixed_wing_tracker.py`,
  `route_intercept_system.py`, `simulate_1hz_test.py` + ilgili testler.
- Aktif çalışma dalı: `test`.
