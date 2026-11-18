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

---

## 🆕 2026-11-17 — Güdüm yeniden tasarımı (canlı simülatörde doğrulandı)

**Ölçülen simülatör gerçekleri** (`analysis/record_telemetry.py`, `analysis/sysid.py`):
- Hedef `rotation` kanalı ~100 Hz, **gecikmesiz ve bozulmasız**: yaw = gerçek uçuş yönü
  (RMS 0.01°), pitch → vz (r=0.99), roll → dönüş hızı (r=0.94). Hedef hızı 1792 ± 23 cm/s.
- Hedef konum: ~5 Hz, **1.0 s gecikmeli**, sabit offset (2,2,1 m), 40 m sıçrama, kesinti.
  Hız kanalı kullanılamaz (r=0.01).
- Dron: eğim = 60°×stick; kararlı hız ≈151 cm/s/°, 3320 cm/s'de doyar; τ≈2 s; yaw 0.5 → ~190°/s.
- ⚠️ Throttle dokümandaki gibi DEĞİL: 0 → +95 cm/s tırmanış; (-0.5, 0) arası tırmandırır
  (-0.05 → +950!); alçalma için -0.6…-1 gerekir. Ters harita: `DroneModel.throttle_for_vz`.

**Yeni modüller:**
- [x] `navigation/talon_estimator.py` — yönelim yol integrali + gecikmeli GPS fit'i
      (`p = b + V·S(t-D)`), D'yi çevrimiçi kestirir (1.006 s buldu), inovasyon kapısı.
      Kayıtta hata 21 m → 3 m (kalan = gözlemlenemez offset). Rota ezberleme YOK (6–20 s pencere).
- [x] `guidance/follow_controller.py` — iz takibi: referans = hedefin d/V s önceki konumu,
      hız + merkezcil ivme feed-forward, fren profilli kapanma, DOLANMA (önden gelince yandan geç),
      3B çarpışma önleme, yaw kamerayı hedefe tutar (pitch/roll ile öteleme ayrık).
- [x] `simulation/follow_runner.py` — tek 50 Hz ana döngü, tek TCP satırı, donma koruması,
      debug truth ile metrik/grafik (truth kontrole GİRMEZ).
- [x] `tests/python/test_follow_offline.py` + `analysis/offline_scenarios.py` — kayıt + dron
      modeliyle simülatörsüz kapalı döngü test.
- [x] `scripts/sim_up.sh` — simülatörü otomatik aç / çökmüşse yeniden başlat.

**Canlı sonuç (follow7, 93 s, takip 20 m):** mesafe medyan 20.0 m, p5–p95 17.0–22.2 m,
en yakın 16.3 m (yaklaşma dahil), hedefi geçme 0, yaw hatası medyan 2.8°.

**Açık işler:**
- [x] `rotation` kanalı yarışmada YOK (şartname) → GPS-only hat ana yol oldu (2026-11-18, aşağıda).
- [ ] Şartnamedeki takip/kilit tanımına göre takip mesafesi ve irtifa ofseti (kamera 25° yukarı).
- [ ] Görü (RF-DETR) ile terminal faz / kilit; kameradan offset kestirimi.
- [x] Simülatör çökmesi — kök neden sim TCP sunucusunda NUL'suz `strlen`; `drone_sdk` komutları `\n\0` ile bitiriyor (2026-11-17).

---

## 🆕 2026-11-18 — Şartnameye uygun yarışma modu + ortam

**Şartname:** hedef = enlem/boylam/irtifa/yer hızı (≤5 Hz); avcı = GPS + yer hızı + duruş
(3.125–50 Hz). Hedef rotation ve dron hız vektörü verilmiyor → 2026-11-17 kestiricisi geçersiz.

- [x] `simulation/competition_telemetry.py` — şartname filtresi (kontrol yolu sadece buradan okur).
- [x] `navigation/gps_target_estimator.py` — GPS-only hedef kestirici (segment offset, kesinti sönümü).
      10 kayıtta iz noktası hatası medyan 3.0 m, p95 8.0 m.
- [x] `navigation/hunter_state.py` — avcı hız vektörü KF; 3.125 Hz'de de canlı sorunsuz.
- [x] `follow_runner.py` varsayılan yarışma modu (`--legacy`, `--drone-hz`); `test_follow_offline.py`
      yarışma modu + `true_hunter` teşhisi.
- [x] Canlı: 40 m takip → 36–45 m bandı, iz hatası 3.5 m; 20 m dönüşlerde açılıyor.
- [x] GPU: proje `.venv` + `nvidia-cudnn-cu12`; RF-DETR 290 ms → ~25 ms (`docs/simulator_ve_ortam.md`).
- [x] Repo düzeni: `analysis/data/` ve `object_detection/*.onnx` git dışı; dokümanlar
      `docs/yarisma_modu_takip.md`, `docs/simulator_ve_ortam.md`.

**Açık işler:**
- [ ] LLA → yerel ENU dönüşümü (`CompetitionTelemetry` girişine).
- [ ] Hedef konum gecikmesi D'nin kalibrasyonu (şu an sabit 1.0 s).
- [ ] Dedektör menzili: arkadan sadece ~10–15 m; uzak menzil için truth ile otomatik etiketli
      sim verisiyle ince ayar. HUD "READY" yazısı yanlış alarm üretiyor → HUD maskesi.
- [ ] Sim bazen bağlantı kapandıktan sonra yeni bağlantı kabul etmiyor.
- [ ] pytest ortamı: sistemdeki anyio eklentisi pytest'i bozuyor (`-p no:anyio`); `tests/python`
      dosyaları pytest testi değil, script.
