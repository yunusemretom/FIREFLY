# Simülatör ve Geliştirme Ortamı

> Durum: 2026-11-18

## 1. Simülatörü çalıştırma (Linux + Wine)

```bash
scripts/sim_up.sh
```

- Simülatör açık ve telemetri akıyorsa hiçbir şey yapmaz.
- Değilse asılı/çökmüş süreçleri (crash reporter dahil) kapatır, `wine` ile başlatır ve
  TCP `:12345` dinlemeye başlayana kadar bekler.
- Ortam değişkenleri: `SIM_DIR` (varsayılan
  `~/Desktop/Drones of War Teknofest/DronesOfWar/Binaries/Win64`), `SIM_LOG`
  (varsayılan `/tmp/dronesofwar_sim.log`), `SIM_TIMEOUT` (120 s).
- "Telemetri akıyor" kontrolü: hedefin yönelimi 0.6 s içinde değişmiyorsa oyun donmuş sayılır.

Wine çıktısındaki `fixme:` ve `libEGL warning` satırları zararsızdır. Oyun D3D11 ile DXVK
üzerinden NVIDIA kartta çalışır.

## 2. Çökme sorunu — çözüldü (2026-11-17)

**Belirti:** Simülatör 20 s ile birkaç dakika arasında rastgele
`EXCEPTION_ACCESS_VIOLATION` ile kapanıyordu.

**Kök neden:** Çökme raporlarındaki (`~/.wine/drive_c/users/<kullanıcı>/Local Settings/Application Data/DronesOfWar/Saved/Crashes/`)
minidump'lar çözümlendi: 8 çökmenin 8'i de aynı komutta (`exe+0x3f29533`). Bu komut,
simülatörün TCP sunucusunda `FSocket::Recv` ile alınan tamponu string'e çeviren satır içi
bir `strlen` döngüsü. Sunucu, gelen baytların `\0` ile bittiğini varsayıyor. SDK ise
komutları sadece `\n` ile bitiriyordu. Tampon bellekte 64 KB'lık bir sayfanın sonuna denk
geldiğinde okuma sayfa dışına taşıp erişim hatası veriyordu. Wine, GPU veya bellekle
ilgisi yok.

**Düzeltme:** `drone_sdk.py` → `send_inputs()` her komutu `\n\0` ile bitiriyor.
15 dk × 100 Hz (~90 bin mesaj) testinde çökme olmadı, komutlar işlendi.

**Asıl düzeltme oyun tarafında olmalı:** Sunucu, string'e çevirirken `BytesRead` uzunluğunu
kullanmalı. Organizasyona bildirilmesi önerilir.

**Bilinen ayrı sorun:** Bir istemci bağlantıyı kapattıktan sonra sim bazen yeni bağlantı
kabul etmiyor (port dinlemeyi bırakıyor). `sim_up.sh` bu durumda sim'i yeniden başlatır.

## 3. GPU ile nesne tespiti (proje `.venv`'i)

Sistem Python ortamı GPU için bozuk:
- `onnxruntime` (CPU) ve `onnxruntime-gpu` aynı yerde kurulu; CPU paketi GPU paketini gölgeliyor.
- `~/.local/.../nvidia/cudnn` üzerine CUDA 13 sürümü yazılmış. Sürücü 550 en fazla
  **CUDA 12.4** desteklediği için cuDNN başlatılamıyor. torch da CUDA 13 için kurulu.

Çözüm: sistem paketlerine dokunmadan proje sanal ortamı.

```bash
python3 -m venv .venv
.venv/bin/pip install --no-cache-dir -r requirements.txt matplotlib "onnxruntime-gpu==1.23.2"
.venv/bin/pip install --no-cache-dir --no-deps "nvidia-cudnn-cu12>=9,<10"
```

- cuBLAS / cudart / cuFFT / cuRAND için sistemde zaten kurulu CUDA 12 kütüphaneleri
  (`~/.local/lib/python3.10/site-packages/nvidia/*`) kullanılır.
- `TalonDetector` bu kütüphaneleri mutlak yoldan kendisi yükler (önce `.venv`'deki cuDNN).
  `LD_LIBRARY_PATH` gerekmez. GPU açılamazsa uyarı verip CPU'ya düşer.
- Sonuç: çıkarım **290 ms (CPU) → ~25 ms (RTX 4050)**.
- Kamera ile çalışırken `.venv/bin/python` kullanın.

> Disk neredeyse dolu (~2 GB boş). Kurulumlarda `--no-cache-dir` kullanın. pip önbelleği
> (`~/.cache/pip`, ~4 GB) `pip cache purge` ile temizlenebilir.

## 4. Kamera ve dedektör — ölçülen durum

- Kamera modeli (`vision/camera_model.py`): yatay FOV 125°, 25° yukarı eğik, 1600×907
  (siyah bantlar kırpılmış). Yakın mesafede projeksiyon hatası ~5 px. Kare ile poz
  arasındaki zaman kayması ≈ 0.
- Dedektör (`vision/talon_detector.py`, RF-DETR ONNX, 512 px ROI):
  - 7–8 m arkadan: Talon ~95 px, skor ~0.8.
  - 25–40 m arkadan: kanatlar kenardan görünüyor, hareket bulanıklığı ve düşük kontrastla
    ~10–15 px soluk siluet. Mevcut model neredeyse hiç bulamıyor. ROI büyütme fayda etmedi.
  - Yanlış alarm: HUD'daki "READY" yazısının "Y" harfi (~25 px), hedefin aynı irtifada
    göründüğü yerin hemen altında. HUD bölgeleri maskelenmeli.
- Uzak menzil için: sim'den gerçek konum (debug truth) ile otomatik etiketlenmiş karelerle
  modele ince ayar gerekir.

## 5. Veri kaydı

```bash
# Ham telemetri + debug truth kaydı
python3 analysis/record_telemetry.py

# Takip koşusu + kare kaydı (kare başına poz ve gerçek konum, meta.jsonl)
.venv/bin/python -m src.python.simulation.follow_runner --seconds 70 --follow 3000 \
    --frames analysis/data/frames3 --frame-period 0.2
```

`analysis/data/` büyük olduğu için git dışındadır (`.gitignore`).
