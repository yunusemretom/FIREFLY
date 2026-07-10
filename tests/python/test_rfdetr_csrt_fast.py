"""
RF-DETR (SAHI'siz, hizlandirilmis) + SORT-CSRT Video Tracking Testi
====================================================================

test_rfdetr_sahi_csrt.py'nin SAHI'siz ve hizlandirilmis versiyonu.

Pipeline:
  Giris (Video dosyasi VEYA canli ekran goruntusu)
    → RF-DETR (tek kare, tam frame) → SORT-CSRT Tracker → Annotated Output

Giris kaynagi:
  INPUT_SOURCE = "screen" → mss ile canli ekran yakalama (X11, ~5-10 ms/kare).
                            SCREEN_MONITOR / SCREEN_REGION ile alan secilir.
  INPUT_SOURCE = "video"  → VIDEO_PATH'teki dosyadan okur (orijinal davranis).

Orijinalden farklar (hiz optimizasyonlari):
  1. SAHI dilimleme KALDIRILDI → kare basina tek inference cagrisi.
     DETR mimarisi NMS gerektirmez; tile birlestirme NMS'i de kalkti.
  2. FP16 + torch.jit.trace derlemesi (optimize_for_inference) + isinma turlari.
  3. On-isleme GPU'da: kare uint8 olarak pinned buffer uzerinden GPU'ya
     yuklenir, float donusumu, resize ve normalize GPU'da yapilir.
     (CPU'da PIL/to_tensor yolu tamamen atlanir.)
  4. predict() BYPASS: derlenmis inference_model + postprocess dogrudan
     cagrilir → kare basina ~5-8 ms Python ek yuku (girdi kontrolleri,
     kaynak kare kopyalari, supervision nesneleri) ortadan kalkar.
     Sorun olursa otomatik predict() fallback devreye girer.
  5. torch.inference_mode() → autograd takibi kapali.
  6. cuDNN benchmark + TF32 matmul aktif.
  7. Video yazimi arka plan thread'inde → disk I/O ana donguyu bloklamaz.
  8. ENABLE_DISPLAY=False ile imshow/waitKey maliyeti de sifirlanabilir.

Faz Sistemi (orijinalle ayni):
  TESPIT FAZI  (DETECT_DURATION sn)
    └─ RF-DETR her kareyi analiz eder (tam kare, dilimsiz).
    └─ Tespitler SORT-CSRT'ye beslenir, tracker'lar olusur/guncellenir.
    └─ Sure dolunca ve en az 1 tracker varsa → TAKIP fazina gecer.

  TAKIP FAZI   (TRACK_DURATION sn, varsayilan 4 sn)
    └─ RF-DETR CALISMAZ → yuksek FPS kazanci.
    └─ SORT-CSRT, CSRT prediction ile nesneleri takip eder.
    └─ Tum tracker'lar kaybolursa erken TESPIT fazina doner.
    └─ Sure dolunca TESPIT fazina doner.

Kullanim:
  cd /home/tom/Documents/FIREFLY
  python3 tests/python/test_rfdetr_csrt_fast.py

Kontroller:
  Q : Cikis
  R : Mevcut fazi sifirla (TESPIT fazina don)

Gereksinimler:
  - rfdetr, opencv-python, numpy, torch, scipy (veya lap)
  - mss (INPUT_SOURCE="screen" icin ekran yakalama)
  - sort_csrt_tracking reposu src/python/vision/ altinda klonlanmis olmali
"""

from __future__ import annotations

import io as _io
import queue
import sys
import threading
import time
import types
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ─── Proje kok dizinini Python yoluna ekle ────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]           # .../FIREFLY
sys.path.insert(0, str(_ROOT))

# sort_csrt.py `from skimage import io` kullaniyor ama Sort/CSRTTracker icin
# gerekli degil. skimage kurulu degilse sahte modul ile import hatasini atla.
if "skimage" not in sys.modules:
    try:
        import skimage  # noqa: F401
    except ImportError:
        _sk = types.ModuleType("skimage")
        _sk.io = types.ModuleType("skimage.io")        # type: ignore[attr-defined]
        sys.modules["skimage"] = _sk
        sys.modules["skimage.io"] = _sk.io              # type: ignore[attr-defined]

_SORT_DIR = _ROOT / "src" / "python" / "vision" / "sort_csrt_tracking"
sys.path.insert(0, str(_SORT_DIR))

from sort_csrt import Sort, CSRTTracker  # noqa: E402

import torch                                           # noqa: E402
from rfdetr import RFDETRSmall                         # noqa: E402
from rfdetr.assets.coco_classes import COCO_CLASSES    # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
#  YAPILANDIRMA — Bu degiskenleri dogrudan degistirerek ayarlayabilirsiniz
# ═══════════════════════════════════════════════════════════════════════════════

# ── Giris / Model ─────────────────────────────────────────────────────────────
# Giris kaynagi: "video" = VIDEO_PATH'ten oku, "screen" = canli ekran goruntusu
INPUT_SOURCE    = "screen"
VIDEO_PATH      = "/home/tom/2026-07-10 00-12-13.mkv"
MODEL_WEIGHTS   = "/home/tom/Downloads/checkpoint_best_regular (09.08.16).pth"
CONF_THRESHOLD  = 0.6
TARGET_FRAME_SIZE: tuple[int, int] | None = None  # (genislik, yukseklik) veya None

# ── Ekran yakalama (INPUT_SOURCE = "screen") ──────────────────────────────────
SCREEN_MONITOR = 1     # mss monitor indeksi (1 = birincil ekran, 0 = tum ekranlar)
# Ekranin sadece bir bolgesini yakala: (left, top, width, height) veya None (tum ekran).
# NOT: ENABLE_DISPLAY acikken onizleme penceresi yakalanan ekrandaysa
# goruntu icinde goruntu (feedback) olusur — pencereyi baska monitore tasiyin
# veya SCREEN_REGION ile pencereyi dislayan bir bolge secin.
SCREEN_REGION: Optional[tuple[int, int, int, int]] = None

# Model giris cozunurlugu. None = checkpoint'in varsayilani (RFDETRSmall: 512).
# Dusurmek (or. 384) inference'i belirgin hizlandirir ama kucuk hedef dogrulugunu
# dusurebilir. patch_size * num_windows katlari olmali (Small icin 64'un kati).
MODEL_RESOLUTION: Optional[int] = None

# ── Hiz / Cikti anahtarlari ───────────────────────────────────────────────────
ENABLE_DISPLAY = True    # False → imshow/waitKey yok, maksimum islem hizi
WRITE_VIDEO    = True    # False → video kaydi yok
WARMUP_FRAMES  = 3       # Ilk kare gecikmesini onlemek icin isinma turu sayisi

# ── Faz sureleri (saniye) ─────────────────────────────────────────────────────
DETECT_DURATION = 1.5    # RF-DETR ile tespit fazi suresi
TRACK_DURATION  = 4.0    # CSRT takip fazi suresi (istediginiz gibi degistirin)

# ── SORT-CSRT parametreleri ───────────────────────────────────────────────────
SORT_MAX_AGE    = 300    # Takip fazinda tracker'in hayatta kalma suresi (kare)
SORT_MIN_HITS   = 1      # Takip baslamadan once gereken min ardisik tespit
SORT_IOU_THRESH = -0.8   # GIoU esigi (negatif = daha esnek esleme)

# ── Tespit disla bolgeleri ────────────────────────────────────────────────────
# Tespit (RF-DETR) YAPILMAYACAK dikdortgen bolgeler: (x1, y1, x2, y2) piksel.
# TAKIP (CSRT) bu bolgelerde calismaya DEVAM EDER.
# Bos liste = maskeleme yok.
DETECTION_EXCLUSION_ZONES: list[tuple[int, int, int, int]] = [(0, 670, 480, 850), (1400, 670, 1920, 850)]
    # (x1, y1, x2, y2),


# ── Cikti ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_FPS = 30.0

# ── Renkler (BGR) ─────────────────────────────────────────────────────────────
COL_DETECT = (0, 220, 90)     # yesil  — tespit kutusu
COL_TRACK  = (0, 140, 255)    # turuncu — takip kutusu
COL_LOST   = (0, 0, 220)      # kirmizi — kayip
COL_TEXT   = (255, 255, 255)
COL_BG     = (20, 20, 20)


# ═══════════════════════════════════════════════════════════════════════════════
#  TESPIT DISLAMA BOLGESI FILTRELEME
# ═══════════════════════════════════════════════════════════════════════════════

def filter_detections_by_exclusion(
    xyxy: np.ndarray,
    confs: np.ndarray,
    class_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tespit dislama bolgelerindeki tespitleri filtreler.
    Bbox merkezi herhangi bir dislama bolgesinde ise tespit atilir.
    Takip (CSRT) etkilenmez — sadece tespit icin kullanilir.
    """
    if not DETECTION_EXCLUSION_ZONES or len(xyxy) == 0:
        return xyxy, confs, class_ids

    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0

    # Hicbir dislama bolgesinde OLMAYAN tespitleri tut
    keep = np.ones(len(xyxy), dtype=bool)
    for zone in DETECTION_EXCLUSION_ZONES:
        in_zone = (
            (cx >= zone[0]) & (cx <= zone[2]) &
            (cy >= zone[1]) & (cy <= zone[3])
        )
        keep &= ~in_zone

    return xyxy[keep], confs[keep], class_ids[keep]


def draw_exclusion_zones(frame: np.ndarray) -> None:
    """Dislama bolgelerini yari-seffaf kirmizi overlay ile gosterir."""
    if not DETECTION_EXCLUSION_ZONES:
        return
    overlay = frame.copy()
    for x1, y1, x2, y2 in DETECTION_EXCLUSION_ZONES:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 80), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)


# ═══════════════════════════════════════════════════════════════════════════════
#  HIZLI TEK-KARE TESPIT (SAHI YOK)
# ═══════════════════════════════════════════════════════════════════════════════

_DEVICE: torch.device = torch.device("cpu")
_USE_CUDA = False
_pinned_buf: Optional[torch.Tensor] = None

# init_fast_inference() tarafindan doldurulur (direct-path onbellekleri)
_FI_MEAN: Optional[torch.Tensor] = None      # (1,3,1,1) normalize ortalamasi
_FI_STD: Optional[torch.Tensor] = None       # (1,3,1,1) normalize std
_FI_RES: int = 0                             # model giris cozunurlugu
_FI_DTYPE: torch.dtype = torch.float32      # optimize edilmis model dtype'i
_FI_READY = False                            # direct path kullanilabilir mi
_EMPTY_DETS = (
    np.empty((0, 4), dtype=np.float32),
    np.empty(0, dtype=np.float32),
    np.empty(0, dtype=np.int32),
)


def init_fast_inference(model: RFDETRSmall) -> None:
    """
    predict()'i bypass eden dogrudan inference yolu icin onbellekleri hazirlar.

    predict() her cagrida ~5-8 ms saf Python ek yuku tasir (PIL/to_tensor
    kontrolleri, kaynak kare kopyalari, supervision nesnesi kurulumu).
    Direct path bu yuku atar: normalize sabitleri GPU'da hazir tutulur,
    derlenmis inference_model dogrudan cagrilir, postprocess elle yapilir.
    """
    global _FI_MEAN, _FI_STD, _FI_RES, _FI_DTYPE, _FI_READY
    try:
        _FI_MEAN = torch.tensor(
            model.means, device=_DEVICE, dtype=torch.float32
        ).view(1, 3, 1, 1)
        _FI_STD = torch.tensor(
            model.stds, device=_DEVICE, dtype=torch.float32
        ).view(1, 3, 1, 1)
        _FI_RES = int(model.model.resolution)
        _FI_DTYPE = model._optimized_dtype or torch.float32
        # Direct path derlenmis inference_model + postprocess gerektirir
        _FI_READY = (
            model._is_optimized_for_inference
            and model.model.inference_model is not None
            and hasattr(model.model, "postprocess")
        )
    except Exception as e:
        print(f"[WARN] Direct inference yolu hazirlanamadi ({e}); "
              f"predict() fallback kullanilacak.")
        _FI_READY = False


def _frame_to_gpu_tensor(frame: np.ndarray) -> torch.Tensor:
    """
    BGR uint8 kareyi (H,W,3) GPU'da (3,H,W) float [0,1] tensore cevirir.

    uint8 upload (float32'nin 1/4'u kadar veri) + pinned buffer ile async
    kopya; float donusumu ve /255 GPU'da yapilir. Kanal sirasi orijinal
    testle ayni birakilir (orijinal de BGR besliyordu → ayni model girdisi).
    """
    global _pinned_buf
    src = torch.from_numpy(frame)
    if _USE_CUDA:
        if _pinned_buf is None or tuple(_pinned_buf.shape) != frame.shape:
            _pinned_buf = torch.empty(
                frame.shape, dtype=torch.uint8, pin_memory=True
            )
        _pinned_buf.copy_(src)
        img = _pinned_buf.to(_DEVICE, non_blocking=True)
    else:
        img = src
    return img.permute(2, 0, 1).float().div_(255.0)


def _predict_fallback(
    model: RFDETRSmall,
    img: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direct path kullanilamiyorsa resmi predict() API'si ile tespit."""
    dets = model.predict(
        img,
        threshold=CONF_THRESHOLD,
        include_source_image=False,
    )
    if len(dets) == 0:
        return _EMPTY_DETS
    return (
        dets.xyxy.astype(np.float32),
        dets.confidence.astype(np.float32),
        dets.class_id.astype(np.int32),
    )


def fast_detect(
    model: RFDETRSmall,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tam kare, tek cagri RF-DETR inference (SAHI'siz, predict() bypass'li).

    DETR NMS-free oldugu ve tile birlestirme olmadigi icin ek NMS gerekmez.
    On-isleme tamamen GPU'da (resize + normalize), derlenmis FP16 model
    dogrudan cagrilir, kutular postprocess ile kare koordinatlarina
    olceklenir. CPU'ya tek toplu kopya yapilir.

    Returns
    -------
    xyxy : (N, 4)  float32
    confs : (N,)   float32
    class_ids : (N,) int32
    """
    h, w = image.shape[:2]
    with torch.inference_mode():
        img = _frame_to_gpu_tensor(image)

        if not _FI_READY:
            return _predict_fallback(model, img)

        # Resize + normalize (predict ile ayni sira: once resize, sonra norm)
        x = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(_FI_RES, _FI_RES),
            mode="bilinear",
            antialias=True,
        )
        x = (x - _FI_MEAN) / _FI_STD
        x = x.to(dtype=_FI_DTYPE)

        preds = model.model.inference_model(x)
        if isinstance(preds, tuple):
            preds = {"pred_logits": preds[1], "pred_boxes": preds[0]}

        target_sizes = torch.tensor([[h, w]], device=_DEVICE)
        result = model.model.postprocess(preds, target_sizes=target_sizes)[0]

        scores = result["scores"]
        keep = scores > CONF_THRESHOLD
        if not bool(keep.any()):
            return _EMPTY_DETS

        # Tek CPU transferi icin paketle
        boxes = result["boxes"][keep].float().cpu().numpy()
        confs = scores[keep].float().cpu().numpy()
        labels = result["labels"][keep].cpu().numpy()

    return (
        boxes.astype(np.float32, copy=False),
        confs.astype(np.float32, copy=False),
        labels.astype(np.int32, copy=False),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  EKRAN YAKALAMA KAYNAGI (cv2.VideoCapture ile ayni arayuz)
# ═══════════════════════════════════════════════════════════════════════════════

class ScreenSource:
    """
    mss ile canli ekran goruntusunu kare kaynagi olarak sunar.

    cv2.VideoCapture gibi read()/release() arayuzu sağlar; ana dongu
    degismeden her iki kaynakla da calisir. mss X11'de GPU'suz ~5-10 ms'de
    tam ekran yakalar. Kareler BGRA gelir, BGR'ye cevrilir (pipeline'in
    beklentisi orijinal testle ayni: BGR uint8).
    """

    def __init__(self) -> None:
        import mss  # yalniz screen modunda gereklidir

        self._sct = mss.mss()
        if SCREEN_REGION is not None:
            left, top, width, height = SCREEN_REGION
            self._mon = {"left": left, "top": top, "width": width, "height": height}
        else:
            monitors = self._sct.monitors
            if not (0 <= SCREEN_MONITOR < len(monitors)):
                raise RuntimeError(
                    f"SCREEN_MONITOR={SCREEN_MONITOR} gecersiz "
                    f"(mevcut: 0..{len(monitors) - 1})"
                )
            self._mon = monitors[SCREEN_MONITOR]

        # HiDPI olcekleme grab boyutunu degistirebilir → gercek boyutu
        # ilk kareden ogren
        ok, probe = self.read()
        if not ok:
            raise RuntimeError("Ekran goruntusu alinamadi (mss.grab basarisiz).")
        self.height, self.width = probe.shape[:2]

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        try:
            shot = self._sct.grab(self._mon)
        except Exception as e:
            print(f"[WARN] Ekran yakalama hatasi: {e}")
            return False, None
        frame = cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)
        return True, frame

    def release(self) -> None:
        self._sct.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  ARKA PLAN VIDEO YAZICI (disk I/O ana donguyu bloklamasin)
# ═══════════════════════════════════════════════════════════════════════════════

class AsyncVideoWriter:
    """cv2.VideoWriter'i ayri thread'de calistirir; write() sadece kuyruga atar."""

    def __init__(self, writer: cv2.VideoWriter, max_queue: int = 64) -> None:
        self._writer = writer
        self._q: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            frame = self._q.get()
            if frame is None:
                break
            self._writer.write(frame)

    def write(self, frame: np.ndarray) -> None:
        # Kuyruk dolarsa (disk cok yavas) bloklar — kare kaybi olmaz.
        self._q.put(frame.copy())

    def release(self) -> None:
        self._q.put(None)
        self._thread.join()
        self._writer.release()


# ═══════════════════════════════════════════════════════════════════════════════
#  SORT SESSIZ CAGRI (debug printlerini bastirir)
# ═══════════════════════════════════════════════════════════════════════════════

def _quiet_sort_update(
    tracker: Sort,
    frame: np.ndarray,
    dets: np.ndarray,
) -> np.ndarray:
    """sort_csrt.py icindeki debug printlerini bastirarak Sort.update cagirir."""
    old_stdout = sys.stdout
    sys.stdout = _io.StringIO()
    try:
        return tracker.update(frame, dets)
    finally:
        sys.stdout = old_stdout


# ═══════════════════════════════════════════════════════════════════════════════
#  GORSELLESTIME YARDIMLARI
# ═══════════════════════════════════════════════════════════════════════════════

# Tracker ID'ye gore renk paleti (canli renkler, BGR)
_ID_PALETTE = [
    (255, 100, 50),    # acik mavi
    (50, 200, 50),     # yesil
    (50, 80, 230),     # kirmizi
    (200, 200, 50),    # cyan
    (200, 50, 200),    # magenta
    (50, 200, 200),    # sari
    (255, 150, 0),     # mavi
    (0, 150, 255),     # turuncu
    (150, 255, 100),   # acik yesil
    (100, 50, 255),    # koyu kirmizi
]


def color_for_id(track_id: int) -> tuple:
    """Tracker ID'ye gore sabit, canli renk doner."""
    return _ID_PALETTE[int(track_id) % len(_ID_PALETTE)]


def put_text_bg(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    scale: float = 0.65,
    thickness: int = 1,
    fg: tuple = COL_TEXT,
    bg: tuple = COL_BG,
) -> None:
    """Arka planli metin cizer (okunabilirlik icin)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 2, y - th - bl - 2), (x + tw + 2, y + bl), bg, -1)
    cv2.putText(img, text, (x, y - bl), font, scale, fg, thickness, cv2.LINE_AA)


def draw_timer_bar(
    frame: np.ndarray,
    elapsed: float,
    total: float,
    label: str,
    color: tuple,
) -> None:
    """Ekranin ustune yatay ilerleme cubugu cizer."""
    h, w = frame.shape[:2]
    bar_h  = 8
    fill_w = int(w * min(elapsed / max(total, 1e-6), 1.0))
    cv2.rectangle(frame, (0, 0), (w, bar_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, 0), (fill_w, bar_h), color, -1)
    remaining = max(0.0, total - elapsed)
    put_text_bg(
        frame,
        f"{label} — kalan {remaining:.1f}s",
        (10, bar_h + 18),
        scale=0.6,
        fg=color,
    )


def draw_hud(
    frame: np.ndarray,
    fps: float,
    phase: str,
    n_tracked: int,
    n_detect: int,
) -> None:
    """Ekrana FPS, faz bilgisi ve istatistik yazar."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        (f"FPS: {fps:.1f}",          (10, 55),  0.9, (0, 255, 0)),
        (f"Takip: {n_tracked}",       (10, 85),  0.7, (0, 255, 255)),
        (f"Tespit: {n_detect}",       (10, 112), 0.7, (0, 200, 200)),
        (f"Faz: {phase}",             (10, 139), 0.7, (255, 255, 0)),
    ]
    for text, pos, scale, color in lines:
        cv2.putText(frame, text, pos, font, scale, color, 2, cv2.LINE_AA)

    # Kontrol yardimi
    put_text_bg(frame, "Q: Cikis | R: Sifirla", (w - 210, h - 15), scale=0.5)


def draw_tracked_objects(
    frame: np.ndarray,
    tracked: np.ndarray,
    track_class_map: dict[int, int],
) -> None:
    """
    SORT-CSRT ciktisindaki takip edilen nesneleri cizer.

    tracked format: [[x1,y1,x2,y2,conf,class,track_id], ...]  (N,7)
    """
    if len(tracked) == 0:
        return

    for row in tracked:
        x1, y1   = int(row[0]), int(row[1])
        x2, y2   = int(row[2]), int(row[3])
        conf     = float(row[4])
        cls_id   = int(row[5])
        track_id = int(row[6])

        # Sinif bilgisini guncelle / sakla
        # (SORT unmatched tracker'larda class_id=0 doner, bu yuzden haritada saklariz)
        if cls_id > 0:
            track_class_map[track_id] = cls_id
        else:
            cls_id = track_class_map.get(track_id, -1)

        cls_name = (
            COCO_CLASSES[cls_id] if 0 <= cls_id < len(COCO_CLASSES) else "?"
        )

        color = color_for_id(track_id)

        # Bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Merkez noktasi ve crosshair
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.drawMarker(
            frame, (cx, cy), color,
            cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA,
        )

        # Etiket
        label = f"ID:{track_id} {cls_name} {conf:.2f}"
        put_text_bg(frame, label, (x1, y1 - 4), scale=0.55, fg=color)


# ═══════════════════════════════════════════════════════════════════════════════
#  ANA DONGU
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    global _DEVICE, _USE_CUDA

    # ── Backend hiz ayarlari ──────────────────────────────────────────────────
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # ── Model yukleme ─────────────────────────────────────────────────────────
    print("[INFO] RF-DETR modeli yukleniyor...")
    model_kwargs: dict = {"pretrain_weights": MODEL_WEIGHTS}
    if MODEL_RESOLUTION is not None:
        model_kwargs["resolution"] = MODEL_RESOLUTION
    model = RFDETRSmall(**model_kwargs)

    _DEVICE   = model.model.device
    _USE_CUDA = _DEVICE.type == "cuda"
    print(f"[INFO] Cihaz: {_DEVICE}")

    try:
        model.optimize_for_inference(compile=True, batch_size=1, dtype=torch.float16)
        print("[INFO] FP16 + JIT trace optimize_for_inference aktif.")
    except Exception as e:
        print(f"[WARN] FP16 optimizasyonu uygulanamadi: {e}")

    init_fast_inference(model)
    print(
        "[INFO] Inference yolu: "
        + ("direct (predict bypass)" if _FI_READY else "predict() fallback")
    )

    # ── Giris kaynagi acma (video dosyasi veya canli ekran) ──────────────────
    if INPUT_SOURCE == "screen":
        region_txt = f"bolge={SCREEN_REGION}" if SCREEN_REGION else f"monitor={SCREEN_MONITOR}"
        print(f"[INFO] Ekran yakalama baslatiliyor ({region_txt})...")
        cap = ScreenSource()
        frame_w, frame_h = cap.width, cap.height
        if TARGET_FRAME_SIZE is not None:
            frame_w, frame_h = TARGET_FRAME_SIZE
        out_fps = OUTPUT_FPS
        print(f"[INFO] Ekran: {frame_w}x{frame_h} (canli, sonsuz akis)")
    elif INPUT_SOURCE == "video":
        print(f"[INFO] Video aciliyor: {VIDEO_PATH}")
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"Video acilamadi: {VIDEO_PATH}")

        frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if TARGET_FRAME_SIZE is not None:
            frame_w, frame_h = TARGET_FRAME_SIZE
        src_fps      = cap.get(cv2.CAP_PROP_FPS)
        out_fps      = src_fps if src_fps and src_fps > 0 else OUTPUT_FPS
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(
            f"[INFO] Video: {frame_w}x{frame_h} @ {src_fps:.1f} FPS, "
            f"{total_frames} kare"
        )
    else:
        raise ValueError(
            f"INPUT_SOURCE 'video' veya 'screen' olmali, "
            f"'{INPUT_SOURCE}' gecersiz."
        )

    # ── Isinma (ilk kare JIT/cudnn gecikmesini onler) ─────────────────────────
    dummy = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    for _ in range(max(WARMUP_FRAMES, 1)):
        fast_detect(model, dummy)
    print(f"[INFO] Isinma tamam ({WARMUP_FRAMES} tur).")

    # ── SORT-CSRT tracker ─────────────────────────────────────────────────────
    sort_tracker = Sort(
        max_age=SORT_MAX_AGE,
        min_hits=SORT_MIN_HITS,
        iou_threshold=SORT_IOU_THRESH,
    )
    track_class_map: dict[int, int] = {}    # track_id → class_id sakla
    print(
        f"[INFO] SORT-CSRT: max_age={SORT_MAX_AGE}, "
        f"min_hits={SORT_MIN_HITS}, iou_thresh={SORT_IOU_THRESH}"
    )
    print("[INFO] Tespit modu: tam kare, tek inference (SAHI yok).")

    # ── Cikti dizini ve video yazici ──────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_writer: Optional[AsyncVideoWriter] = None
    output_path = ""
    if WRITE_VIDEO:
        writer_candidates = [
            ("fast_csrt_output.mp4", "mp4v"),
            ("fast_csrt_output.avi", "XVID"),
            ("fast_csrt_output.avi", "MJPG"),
        ]
        for fname, codec in writer_candidates:
            candidate = OUTPUT_DIR / fname
            writer = cv2.VideoWriter(
                str(candidate),
                cv2.VideoWriter_fourcc(*codec),
                out_fps,
                (frame_w, frame_h),
            )
            if writer.isOpened():
                video_writer = AsyncVideoWriter(writer)
                output_path  = str(candidate)
                print(f"[INFO] Video kaydi (async): {codec} -> {output_path}")
                break
            writer.release()

        if video_writer is None:
            print("[WARN] Video yazici olusturulamadi. Sadece ekranda gosterilecek.")

    # ── Pencere ───────────────────────────────────────────────────────────────
    win_name = "RF-DETR FAST + SORT-CSRT"
    if ENABLE_DISPLAY:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # ── Durum degiskenleri ────────────────────────────────────────────────────
    prev_time = time.perf_counter()
    fps       = 0.0
    fps_alpha = 0.9

    in_track        = False                # False = TESPIT, True = TAKIP
    phase_start     = time.perf_counter()
    frame_idx       = 0
    n_detect_current = 0                   # mevcut karedeki ham tespit sayisi

    print(
        f"\n{'='*60}\n"
        f"  Faz dongusu: TESPIT({DETECT_DURATION}s) -> TAKIP({TRACK_DURATION}s) -> ...\n"
        f"  Q=Cikis | R=Sifirla\n"
        f"{'='*60}\n"
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if INPUT_SOURCE == "screen":
                    print("\n[INFO] Ekran yakalama durdu.")
                else:
                    print("\n[INFO] Video bitti.")
                break

            if TARGET_FRAME_SIZE is not None:
                frame = cv2.resize(frame, TARGET_FRAME_SIZE)

            frame_idx += 1

            # ── FPS ──────────────────────────────────────────────────────────
            now = time.perf_counter()
            dt  = now - prev_time
            if dt > 0:
                cur_fps = 1.0 / dt
                fps = (
                    fps_alpha * fps + (1.0 - fps_alpha) * cur_fps
                    if fps > 0
                    else cur_fps
                )
            prev_time = now

            elapsed = now - phase_start
            tracked = np.empty((0, 7))

            # ═════════════════════════════════════════════════════════════════
            # TESPIT FAZI: RF-DETR (tam kare) + SORT-CSRT
            # ═════════════════════════════════════════════════════════════════
            if not in_track:
                phase = "TESPIT"

                # Tek kare hizli tespit (SAHI yok)
                xyxy, confs, class_ids = fast_detect(model, frame)

                # Dislama bolgelerindeki tespitleri at (takip etkilenmez)
                xyxy, confs, class_ids = filter_detections_by_exclusion(
                    xyxy, confs, class_ids,
                )
                n_detect_current = len(confs)

                # SORT'a besle: [x1,y1,x2,y2,conf,class_id] (N,6)
                if n_detect_current > 0:
                    dets = np.column_stack([xyxy, confs, class_ids])
                else:
                    dets = np.empty((0, 6))

                tracked = _quiet_sort_update(sort_tracker, frame, dets)

                # Dislama bolgesi overlay
                draw_exclusion_zones(frame)

                # Ham tespitleri ince yesil cerceve ile ciz
                for box in xyxy:
                    bx1, by1 = int(box[0]), int(box[1])
                    bx2, by2 = int(box[2]), int(box[3])
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), COL_DETECT, 1)

                # SORT tracker sonuclarini ciz
                draw_tracked_objects(frame, tracked, track_class_map)

                # Timer cubugu
                draw_timer_bar(frame, elapsed, DETECT_DURATION, "TESPIT", COL_DETECT)

                # Sure doldu → TAKIP fazina gec (en az 1 tracker varsa)
                if elapsed >= DETECT_DURATION:
                    n_tracks = len(tracked)
                    if n_tracks > 0:
                        print(
                            f"[KARE {frame_idx}] TAKIP fazina gecildi — "
                            f"{n_tracks} hedef takipte"
                        )
                        in_track = True
                    else:
                        print(
                            f"[KARE {frame_idx}] Tespit bulunamadi, "
                            f"TESPIT fazi yenilendi."
                        )
                    phase_start = time.perf_counter()

            # ═════════════════════════════════════════════════════════════════
            # TAKIP FAZI: Sadece SORT-CSRT (RF-DETR yok → hizli)
            # ═════════════════════════════════════════════════════════════════
            else:
                phase = "TAKIP"
                n_detect_current = 0

                # SORT'a bos tespit besle — CSRT tahmin yapar
                tracked = _quiet_sort_update(
                    sort_tracker, frame, np.empty((0, 6))
                )
                # Dislama bolgesi overlay (takip filtrelenmez)
                draw_exclusion_zones(frame)

                # Takip edilen nesneleri ciz
                draw_tracked_objects(frame, tracked, track_class_map)

                # Timer cubugu
                draw_timer_bar(frame, elapsed, TRACK_DURATION, "TAKIP", COL_TRACK)

                # Tum tracker'lar kayboldu → erken TESPIT'e don
                if len(tracked) == 0:
                    print(
                        f"[KARE {frame_idx}] Tum hedefler kayboldu — "
                        f"TESPIT fazina donuluyor."
                    )
                    in_track    = False
                    phase_start = time.perf_counter()

                # Sure doldu → TESPIT fazina don
                elif elapsed >= TRACK_DURATION:
                    print(
                        f"[KARE {frame_idx}] Takip suresi doldu ({TRACK_DURATION}s) — "
                        f"TESPIT fazina donuluyor."
                    )
                    in_track    = False
                    phase_start = time.perf_counter()

            # ── HUD ──────────────────────────────────────────────────────────
            draw_hud(frame, fps, phase, len(tracked), n_detect_current)

            # ── Video kayit (arka plan thread) ───────────────────────────────
            if video_writer:
                video_writer.write(frame)

            # ── Ekranda goster ───────────────────────────────────────────────
            if ENABLE_DISPLAY:
                cv2.imshow(win_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    # Manuel sifirlama — SORT'u sifirdan baslat
                    print("[INFO] Manuel sifirlama — TESPIT fazina donuluyor.")
                    in_track = False
                    CSRTTracker.count = 0
                    sort_tracker = Sort(
                        max_age=SORT_MAX_AGE,
                        min_hits=SORT_MIN_HITS,
                        iou_threshold=SORT_IOU_THRESH,
                    )
                    track_class_map.clear()
                    phase_start = time.perf_counter()

    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        if output_path:
            print(f"\n[INFO] Video kaydedildi: {output_path}")
        print(f"[INFO] Toplam {frame_idx} kare islendi.")


if __name__ == "__main__":
    main()
