"""
RF-DETR SAHI + SORT-CSRT Video Tracking Testi
==============================================

Pipeline:
  Video Girisi → SAHI Tiling → RF-DETR → Class-aware NMS → SORT-CSRT Tracker → Annotated Output

Faz Sistemi:
  TESPIT FAZI  (DETECT_DURATION sn)
    └─ SAHI dilimlemeli RF-DETR her kareyi analiz eder.
    └─ Tespitler SORT-CSRT'ye beslenir, tracker'lar olusur/guncellenir.
    └─ Sure dolunca ve en az 1 tracker varsa → TAKIP fazina gecer.

  TAKIP FAZI   (TRACK_DURATION sn, varsayilan 4 sn)
    └─ RF-DETR CALISMAZ → yuksek FPS kazanci.
    └─ SORT-CSRT, CSRT prediction ile nesneleri takip eder.
    └─ Tum tracker'lar kaybolursa erken TESPIT fazina doner.
    └─ Sure dolunca TESPIT fazina doner.

Kullanim:
  cd /home/tom/Documents/FIREFLY
  python3 tests/python/test_rfdetr_sahi_csrt.py

Kontroller:
  Q : Cikis
  R : Mevcut fazi sifirla (TESPIT fazina don)

Gereksinimler:
  - rfdetr, opencv-python, numpy, torch, scipy (veya lap)
  - sort_csrt_tracking reposu src/python/vision/ altinda klonlanmis olmali
"""

from __future__ import annotations

import io as _io
import sys
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
VIDEO_PATH      = "tests/python/outputs/fast_csrt_output.mp4"
MODEL_WEIGHTS   = "/home/tom/Downloads/checkpoint_best_regular (09.08.16).pth"
CONF_THRESHOLD  = 0.7
TARGET_FRAME_SIZE: tuple[int, int] | None = (640,375)  # Ornek: (1280, 720) (genislik, yukseklik) veya None

# ── Faz sureleri (saniye) ─────────────────────────────────────────────────────
DETECT_DURATION = 1.5    # SAHI + RF-DETR ile tespit fazi suresi
TRACK_DURATION  = 4.0    # CSRT takip fazi suresi (istediginiz gibi degistirin)

# ── SAHI dilimleme ────────────────────────────────────────────────────────────
SLICE_WIDTH       = 640
SLICE_HEIGHT      = 640
OVERLAP_RATIO     = 0.2
NMS_IOU_THRESHOLD = 0.5

# ── SORT-CSRT parametreleri ───────────────────────────────────────────────────
SORT_MAX_AGE    = 300    # Takip fazinda tracker'in hayatta kalma suresi (kare)
SORT_MIN_HITS   = 1      # Takip baslamadan once gereken min ardisik tespit
SORT_IOU_THRESH = -0.8   # GIoU esigi (negatif = daha esnek esleme)

# ── Tespit disla bolgeleri ────────────────────────────────────────────────────
# Tespit (RF-DETR) YAPILMAYACAK dikdortgen bolgeler: (x1, y1, x2, y2) piksel.
# TAKIP (CSRT) bu bolgelerde calismaya DEVAM EDER.
# Bos liste = maskeleme yok.
# Ornek: [(0, 0, 300, 1080), (1620, 0, 1920, 1080)]  → sol ve sag kenar
DETECTION_EXCLUSION_ZONES: list[tuple[int, int, int, int]] = [(0, 670, 480,850 ),(1400,670,1920,850) ]
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

def _center_in_zone(
    cx: float, cy: float,
    zone: tuple[int, int, int, int],
) -> bool:
    """Merkez noktasinin zone icinde olup olmadigini doner."""
    return zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]


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
#  SAHI DILIMLEME (run_rfdetr_sahi_test.py'den adapte)
# ═══════════════════════════════════════════════════════════════════════════════

_cached_slices: Optional[list[tuple[int, int, int, int]]] = None
_cached_shape: tuple[int, int] = (-1, -1)


def compute_slices(img_h: int, img_w: int) -> list[tuple[int, int, int, int]]:
    """Tile koordinatlarini (x0, y0, x1, y1) listesi olarak doner."""
    step_x = max(int(SLICE_WIDTH * (1.0 - OVERLAP_RATIO)), 1)
    step_y = max(int(SLICE_HEIGHT * (1.0 - OVERLAP_RATIO)), 1)
    slices: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for y0 in range(0, img_h, step_y):
        y1 = min(y0 + SLICE_HEIGHT, img_h)
        y0 = max(0, y1 - SLICE_HEIGHT)
        for x0 in range(0, img_w, step_x):
            x1  = min(x0 + SLICE_WIDTH, img_w)
            x0_ = max(0, x1 - SLICE_WIDTH)
            key = (x0_, y0, x1, y1)
            if key not in seen:
                seen.add(key)
                slices.append(key)
    return slices


def get_slices(h: int, w: int) -> list[tuple[int, int, int, int]]:
    """Tile listesi frame boyutu degismedikce yeniden hesaplanmaz."""
    global _cached_slices, _cached_shape
    if (h, w) != _cached_shape:
        _cached_slices = compute_slices(h, w)
        _cached_shape  = (h, w)
    return _cached_slices  # type: ignore[return-value]


def class_aware_nms(
    xyxy: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    score_threshold: float,
    iou_threshold: float,
) -> np.ndarray:
    """Sinif-bazli NMS. Korunan indeksleri doner."""
    keep: list[int] = []
    for cls in np.unique(classes):
        idx   = np.where(classes == cls)[0]
        boxes = xyxy[idx]
        s     = scores[idx].tolist()
        xywh  = np.column_stack((
            boxes[:, 0], boxes[:, 1],
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
        )).tolist()
        sel = cv2.dnn.NMSBoxes(xywh, s, score_threshold, iou_threshold)
        if sel is not None and len(sel):
            keep.extend(idx[np.array(sel).reshape(-1).astype(int)].tolist())
    return (
        np.array(sorted(set(keep)), dtype=np.int32)
        if keep
        else np.empty(0, dtype=np.int32)
    )


def sahi_detect(
    model: RFDETRSmall,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SAHI dilimlemeli RF-DETR inference.

    Returns
    -------
    xyxy : (N, 4)  float32
    confs : (N,)   float32
    class_ids : (N,) int32
    """
    h, w   = image.shape[:2]
    slices = get_slices(h, w)

    all_xyxy: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    all_cls:  list[np.ndarray] = []

    for x0, y0, x1, y1 in slices:
        tile      = image[y0:y1, x0:x1]
        tile_dets = model.predict(tile, threshold=CONF_THRESHOLD)
        if len(tile_dets) == 0:
            continue

        tile_xyxy = tile_dets.xyxy.astype(np.float32)
        # Tile koordinatlarini ana kareye donustur
        tile_xyxy[:, [0, 2]] += x0
        tile_xyxy[:, [1, 3]] += y0

        all_xyxy.append(tile_xyxy)
        all_conf.append(tile_dets.confidence.astype(np.float32))
        all_cls.append(tile_dets.class_id.astype(np.int32))

    if not all_xyxy:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )

    xyxy    = np.concatenate(all_xyxy)
    confs   = np.concatenate(all_conf)
    classes = np.concatenate(all_cls)

    keep = class_aware_nms(
        xyxy, confs, classes,
        score_threshold=max(CONF_THRESHOLD * 0.5, 0.01),
        iou_threshold=NMS_IOU_THRESHOLD,
    )

    if keep.size == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )

    return xyxy[keep], confs[keep], classes[keep]


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
        f"{label} \u2014 kalan {remaining:.1f}s",
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
    # ── Model yukleme ─────────────────────────────────────────────────────────
    print("[INFO] RF-DETR modeli yukleniyor...")
    model = RFDETRSmall(pretrain_weights=MODEL_WEIGHTS)

    try:
        model.optimize_for_inference(dtype=torch.float16)
        print("[INFO] FP16 optimize_for_inference aktif.")
    except Exception as e:
        print(f"[WARN] FP16 optimizasyonu uygulanamadi: {e}")

    # ── Video acma ────────────────────────────────────────────────────────────
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

    # ── SAHI bilgisi ──────────────────────────────────────────────────────────
    n_tiles = len(compute_slices(frame_h, frame_w))
    print(
        f"[INFO] SAHI: {SLICE_WIDTH}x{SLICE_HEIGHT} tile, "
        f"overlap={OVERLAP_RATIO}, toplam {n_tiles} tile/kare"
    )

    # ── Cikti dizini ve video yazici ──────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    writer_candidates = [
        ("sahi_csrt_output.mp4", "mp4v"),
        ("sahi_csrt_output.avi", "XVID"),
        ("sahi_csrt_output.avi", "MJPG"),
    ]
    video_writer: Optional[cv2.VideoWriter] = None
    output_path = ""
    for fname, codec in writer_candidates:
        candidate = OUTPUT_DIR / fname
        writer = cv2.VideoWriter(
            str(candidate),
            cv2.VideoWriter_fourcc(*codec),
            out_fps,
            (frame_w, frame_h),
        )
        if writer.isOpened():
            video_writer = writer
            output_path  = str(candidate)
            print(f"[INFO] Video kaydi: {codec} -> {output_path}")
            break
        writer.release()

    if video_writer is None:
        print("[WARN] Video yazici olusturulamadi. Sadece ekranda gosterilecek.")

    # ── Pencere ───────────────────────────────────────────────────────────────
    win_name = "RF-DETR SAHI + SORT-CSRT"
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
            # TESPIT FAZI: SAHI + RF-DETR + SORT-CSRT
            # ═════════════════════════════════════════════════════════════════
            if not in_track:
                phase = "TESPIT"

                # SAHI ile dilimli tespit
                xyxy, confs, class_ids = sahi_detect(model, frame)

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

            # ── Video kayit ──────────────────────────────────────────────────
            if video_writer:
                video_writer.write(frame)

            # ── Ekranda goster ───────────────────────────────────────────────
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
