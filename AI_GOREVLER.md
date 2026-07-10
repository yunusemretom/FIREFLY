<!--
================================================================================
  YAPAY ZEKA ARAÇLARI İÇİN NOT (AI ASSISTANT DIRECTIVE)
  Firefly görev durumunun OTORİTE listesi. Bir görevi tamamladığında/ele
  aldığında bu dosyayı GÜNCELLE (kutuyu işaretle, tarih düş). Kod tabanını
  yeniden analiz etmeden önce buraya bak.
  Kardeş dosyalar: AI_PROJE_OZETI.md, AI_DOSYA_YAPISI.md  |  Güncel: 2026-07-10
================================================================================
-->

# ✅ Firefly — Görevler (Yapılan / Yapılacak)

Efsane: `[x]` tamamlandı · `[~]` kısmen/devam ediyor · `[ ]` bekliyor

---

## 🟢 Tamamlanan Görevler

### Navigasyon / Sensör Füzyonu
- [x] TCP SDK entegrasyonu (`drone_sdk.py`): telemetri parse, kontrol gönderimi,
      debug truth + corruption maskesi çözümleme.
- [x] Config-driven CTRV EKF (`navigation/ekf.py`, 5 durum, analitik Jacobian,
      Mahalanobis spike reddi, dropout tespiti). Parametreler `ekf_params.yaml`.
- [x] **CTRV UKF tracker** (`navigation/fixed_wing_tracker.py`, 7 durum) —
      `hedef_konumlari.xlsx` ile kalibre; 1 Hz'de ham veriye göre ~%27 daha
      düşük hata (5 tohum). `_sanitize_state()` ile fiziksel sınır koruması.
- [x] 1 Hz koşulu performans testi (`simulate_1hz_test.py`): UKF vs CV Kalman.
- [x] Rota tabanlı intercept sistemi (`route_intercept_system.py`): elips fit
      (Fitzgibbon + robust/outlier atma), RoutePlanner (auto-refit),
      `solve_intercept()`. Gerçek veride ~%3.5 kalıntı ile doğrulandı.

### Güdüm / Kontrol
- [x] **Rota takip salınımı/savrulması düzeltildi** (2026-07-10, canlı sim):
      `test_route_intercept.py` canlı takibinde 4 kök neden bulunup giderildi:
      1. `nearest_phi` kuantizasyonu (0.5°≈220 cm carrot zıplaması) → parabolik
         yerel iyileştirme eklendi (hata 0.5°→0.0036°).
      2. `YawController` gyro yolunda sınırsız integral windup + runner'ın
         yanlış alanı sıfırlaması → `int_limit` (150 deg·s) + `reset()` metodu.
      3. Hız (catch) kanalı P-only + LPF gecikmesi → sönümsüz yuva salınımı
         (hedefe 12-20 cm'e kadar tehlikeli dalışlar) → PD'ye çevrildi
         (`catch_damp: 1.2`), halka taban hızı yerel elips hızı v(φ) oldu.
      4. Cross-track band sınırında keskin mod geçişi (132 s'de 26 geçiş →
         70 deg/s yaw patlamaları) → smoothstep ile sürekli harmanlama.
      Ayrıca hız→pitch iç döngüsü limit-cycle'ı (kp 4.0→2.5, kd 0.1→0.35,
      slew 30→45°/s) ve `kd_gyro` 0.002→0.004 (P:D=8.75) düzeltildi.
      → Kilit-sonrası ölçüm (180 s canlı koşu, önce→sonra): yaw RMS 54.7→21.5
        deg/s, savrulma 3.25→0.26 tur/dk, en yakın gerçek mesafe 20 cm
        (neredeyse çarpışma)→1291 cm (güvenli yuva takibi), irtifa RMS ~70 cm.
        Loglar: `tests/python/logs/{baseline,fixed_v1,fixed_v2}_20260710/`.
- [x] PID çekirdeği ve pitch/yaw/throttle kontrolcüleri (`guidance/pid_*.py`).
- [x] Mod geçiş mantığı yeniden yazıldı (dinamik yaw-telafili pitch ölçekleme).
- [x] **Pursuit güdümü v2** (`guidance/pursuit_guidance.py`): canlı moddaki
      dalgalı yol sorunu çözüldü. `MeasurementGate` (stale telemetri fix),
      `LeadPointSmoother`, `SmoothPursuitGuidance` (adaptif lead),
      `YawController` (gyro damping, EMA yok).
      → 10 tohum: heading-rate RMS 31.8→19.1 deg/s, aim p95 18.5k→4.8k,
        yakalama 18.8→17.7 s (10/10 başarılı).
- [x] Canlı-koşul emülasyon harness'i (`test_pursuit_smoothing.py`): stale
      telemetri, faz kayması, dropout, loop jitter; A/B/C karşılaştırma.
- [x] 5 m arkadan hassas takip güdümü (`pid_params.yaml → pursuit`): imzalı
      mesafe hatası + sqrt hız profili.

### Görü (Vision)
- [x] RF-DETR TensorRT/ONNX export araçları (`vision/rfdetr_inference/`).
- [x] RF-DETR inference scriptleri (video, ekran yakalama, SAHI tiling).
- [x] **SAHI + SORT-CSRT video tracking testi** (`tests/python/test_rfdetr_sahi_csrt.py`):
      SAHI dilimlemeli RF-DETR tespit + SORT-CSRT çoklu nesne takibi. Faz bazlı
      döngü: TESPIT (SAHI+RF-DETR) → TAKIP (sadece CSRT, yüksek FPS). (2026-07-09)
- [x] **TRT + SAHI + CSRT + merkez kilitlenme + görsel servolama testi**
      (`tests/python/test_rfdetr_trt_visual_servo.py`, 2026-07-10):
      `run_rfdetr_tensorrt.py` altyapısı (TRTInferencer, sahi_predict,
      ScreenCapture) + SORT-CSRT birleştirildi. Ekran merkezinde histerezisli
      kilitlenme dörtgeni (`LockOnManager`: yapışkan track-ID seçimi, onay/
      bırakma kare sayaçları). **G tuşu** ile açılıp kapanan görsel servolama
      (`VisualServoController`): piksel hatası → FOV (125°) ile açı → LPF
      (tau 0.12 s) → PD + smoothstep devreye-giriş rampası + kanal başına slew
      limit → `drone_sdk.set_control_surfaces` (yaw/throttle/pitch; pitch cos
      hizalama ölçekli). Servo kapalıyken hiçbir komut gönderilmez (sadece
      görüntü işleme + kilit); hedef kaybında komutlar sıfıra söner, 1.5 s
      sonra servo kendini kapatır. Canlı doğrulama bekliyor (simülatör + TRT
      engine gerekli).
      **TAKİP fazı FPS optimizasyonu (2026-07-10):** dört çarpımsal kazanç:
      (1) `TRACK_ONLY_LOCK_TARGET` — faza girerken kilit hedefi dışındaki
      tracker'lar atılır (CSRT maliyeti tracker sayısıyla doğrusal);
      (2) `CSRT_FAST` — sort_csrt fabrikası monkeypatch'le hafifletilmiş
      parametrelere çevrildi (ölçek 33→9, ADMM 4→2, şablon 200→150, kanal
      ağırlıkları kapalı; ~13.5→9 ms, sentetik testte 4.2 ms/kare);
      (3) `TRACK_STRIDE=2` — CSRT her 2 karede bir, arada son kutu;
      (4) yinelenen kare atlama (identity check) + gereksiz `frame.copy()`
      kaldırıldı. Sentetik ölçüm: tek hedef TAKİP güncellemesi ~240 FPS
      (stride öncesi). pycuda ayrıca numpy 2 ile yeniden derlendi
      (`--no-build-isolation`; eski pip izole ortamda numpy 1.x kullanıyordu).
- [x] **GPS→görsel servo YUMUŞAK DEVİR (birleşik test)** (2026-07-10,
      `tests/python/test_rfdetr_route_handover.py`): `test_route_intercept.py`
      (telemetri/GPS rota güdümü) ile `test_rfdetr_trt_visual_servo.py`
      (kamera kilit+servo) birleştirildi. `HandoverManager` durum makinesi:
      GPS_PURSUIT (rota güdümüyle yaklaş) → mesafe < `VISION_ENABLE_CM` (120 m)
      kamera boru hattı çalışır → mesafe < `HANDOVER_CM` (80 m) + kararlı kilit
      (`LOCK_STABLE_S` debounce) → HANDOVER (smoothstep ile `HANDOVER_TIME_S`
      boyunca komut vektörleri harmanlanır) → VISUAL_SERVO. Kilit `RELOCK_GRACE_S`
      histerezisiyle korunur (mod titreşimi giderildi). Kilit+servo çekirdeği
      yeni `src/python/vision/visual_servo.py` modülüne çıkarıldı (ServoConfig +
      LockOnManager + VisualServoController, CUDA'sız); hem görsel servo testi
      hem birleşik test buradan çeker. Canlı modda kamera boru hattı ayrı
      thread'de (`VisionWorker`), 50 Hz kontrol döngüsü telemetriyle GPS güdümü
      koşup blend eder. `--sim` modu donanımsız (sentetik kamera projeksiyonu):
      **doğrulandı** — 90 s koşuda 2 temiz mod geçişi (16.6 s→handover,
      21.9 s→servo), servo fazı piksel hata x_RMS ~60 px (hedef ortalı),
      yaw komut değişimi 0.0024/adım (yumuşak), en yakın 7.7 m.
      Rapor: `tests/python/outputs/route_handover_sim.png`. Canlı doğrulama
      simülatör + TRT engine ile bekliyor.

### GCS / Arayüz
- [x] Web tabanlı yer kontrol istasyonu: `server.js` TCP↔WebSocket köprüsü,
      tarayıcı arayüzü (harita, telemetri, kontroller, enstrümanlar).
- [x] Electron iskeleti + React bileşenleri (renderer/).
- [x] GCS kullanım kılavuzu (`docs/gcs_server_kullanim_kilavuzu.md`).
- [x] Eski Tkinter kontrol paneli + Excel veri toplama (`drone_gui.py`).

### Bakım / Kalite
- [x] Legacy navigasyon modülleri temizlendi (CRSF, gnss_filter, waypoint_controller,
      nav_integration, gnss_jammer, scenario_runner, eski bridge/test'ler).
- [x] Ruff ile kod kalitesi iyileştirmeleri, syntax düzeltmeleri.

---

## 🟡 Devam Eden / Kısmi

- [~] **Canlı doğrulama** — pursuit v2 emülasyonda başarılı; kullanıcı gerçek
      simülatörde `tests/python/test_pursuit_live.py` ile doğrulayacak.
- [~] Görü modülü çekirdeği (`vision/detector.py`, `tracker.py`,
      `lock_detector.py`) hâlâ iskelet — inference araçları hazır ama ana
      pipeline'a (kilitlenme→güdüm) bağlı değil.
- [~] `comms/` ve `simulation/sim_bridge.py` modülleri iskelet halinde.
- [~] EKF vs UKF: iki filtre de mevcut; hangisinin ana yol olacağı netleşmeli
      (şu an UKF tracker aktif kullanılıyor).

---

## 🔴 Yapılacak / Açık İşler

### Yüksek Öncelik
- [ ] **Terminal faz taşma turları:** drone hedeften hızlıyken yakın mesafede
      orbit atıyor (aşırı fırlama). Ayrı bir problem; gerekirse **proportional
      navigation (PN)** güdümü ekle. (Kaynak: pursuit v2 notları.)
- [ ] Canlı test sonuçlarını AI_PROJE_OZETI.md'ye işleyip parametreleri
      son-kalibre et (kd, lead_time, follow_distance).
- [ ] Rota-intercept sistemini (`route_intercept_system.py`) canlı pursuit
      döngüsüne entegre et — şu an bağımsız modül.

### Orta Öncelik
- [ ] Görü pipeline'ını kapat: RF-DETR tespit → kilit → güdüm devri (görsel
      tabanlı terminal güdüm). `vision/*` iskeletlerini gerçekle.
- [ ] `comms/` (server_client, telemetry_parser, data_broker) modüllerini
      yarışma sunucusu (`competition_server.yaml`) ile gerçekle.
- [ ] EKF vs UKF için tek bir "ana filtre" seç, diğerini kıyas testine indir.

### Düşük Öncelik / Bakım
- [ ] Kök `README.md`'yi güncelle (silinmiş modül atıflarını temizle, yeni
      pursuit/tracker yolunu yansıt).
- [ ] `docs/` altındaki eskimiş rehberleri (sim_bridge_ekf_rehberi vb.) güncelle
      veya arşivle.
- [ ] `mode_manager.py`'yi pursuit v2 + PN mod geçişleriyle tamamla.
- [ ] Çalışan artefaktları temizle (`.~lock.hedef_konumlari.xlsx#`).

---

## Notlar
- Görev kaynakları: git geçmişi, kod başlıkları, proje hafıza notları
  (pursuit-guidance-v2, ekf-navigation, project-overview).
- Bir görevi kapattığında: kutuyu `[x]` yap, tarihi güncelle, gerekirse
  AI_PROJE_OZETI.md'deki ilgili bölümü de düzelt.
