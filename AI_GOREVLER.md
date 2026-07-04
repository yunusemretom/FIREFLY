<!--
================================================================================
  YAPAY ZEKA ARAÇLARI İÇİN NOT (AI ASSISTANT DIRECTIVE)
  Firefly görev durumunun OTORİTE listesi. Bir görevi tamamladığında/ele
  aldığında bu dosyayı GÜNCELLE (kutuyu işaretle, tarih düş). Kod tabanını
  yeniden analiz etmeden önce buraya bak.
  Kardeş dosyalar: AI_PROJE_OZETI.md, AI_DOSYA_YAPISI.md  |  Güncel: 2026-07-04
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
