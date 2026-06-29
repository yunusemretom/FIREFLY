# Firefly Proje Dokümantasyonu

## Rehberler

| Dosya | Konu | İçerik |
|---|---|---|
| [ekf_rehberi.md](ekf_rehberi.md) | EKF Hedef Filtresi | Durum vektörü, spike reddi, dropout tespiti, config parametreleri |
| [pursuit_rehberi.md](pursuit_rehberi.md) | Pursuit Güdüm | Takip noktası, imzalı mesafe, dış PID, spin önleme |
| [sim_bridge_ekf_rehberi.md](sim_bridge_ekf_rehberi.md) | Test Bridge'leri | Konum öncelik sırası, agresif kovalama vs hassas takip |
| [pid_ve_sim_bridge_rehberi.md](pid_ve_sim_bridge_rehberi.md) | PID Guidance | pid_core, pid_pitch, pid_yaw, pid_throttle, sim_bridge |
| [gcs_server_kullanim_kilavuzu.md](gcs_server_kullanim_kilavuzu.md) | GCS Web Sunucu | Tarayıcı tabanlı yer kontrol istasyonu |
| [architecture_diagrams.md](architecture_diagrams.md) | Mimari | Sistem bileşenleri ve veri akışı diyagramları |
| [competition_notes.md](competition_notes.md) | Yarışma Notları | Yarışmaya özel kurallar ve gözlemler |

## Hızlı Başlangıç

```bash
# Agresif kovalama (hedefe maksimum hızda yaklaş):
python3 tests/python/sim_bridge_ekf.py

# 5 m arkadan hassas takip:
python3 tests/python/sim_bridge_pursuit.py

# EKF canlı izleme (grafik + terminal):
python3 tests/python/navigation/live_ekf_monitor.py
```

## Mimari Özeti

```
drone_sdk  ──►  _best_target_pos()  ──►  Guidance
  (GPS)         Truth > EKF > Raw        │
                                         ├── sim_bridge_ekf.py    → agresif kovalama
                                         └── sim_bridge_pursuit.py → 5 m takip
                                                    │
                                              PursuitController
                                                    │
                                         ┌──────────┴──────────┐
                                      Throttle             Pitch + Yaw
                                     (irtifa PID)        (mesafe PID → hız PID)
```

## Temel Dosya Haritası

```
config/
  pid_params.yaml       ← tüm PID, limit, threshold, pursuit parametreleri
  ekf_params.yaml       ← EKF gürültü ve kovaryans parametreleri

src/python/
  navigation/ekf.py             ← TargetEKF sınıfı
  guidance/pursuit.py           ← PursuitController sınıfı
  guidance/pid_core.py          ← PID, SlewLimiter, clamp, wrap_180
  guidance/pid_pitch.py         ← PitchController (orijinal sim_bridge için)
  guidance/pid_throttle.py      ← ThrottleController
  guidance/pid_yaw.py           ← YawController
  simulation/sim_bridge.py      ← orijinal bridge (dokunulmaz)

tests/python/
  sim_bridge_ekf.py             ← agresif kovalama test bridge'i
  sim_bridge_pursuit.py         ← 5 m takip test bridge'i
  navigation/live_ekf_monitor.py ← EKF canlı izleme
```
