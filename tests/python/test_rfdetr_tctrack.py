"""
RF-DETR TensorRT + TCTrack — Otomatik Degisim Dongusu
======================================================

Akis:
  TESPIT FAZI  (4 sn)
    └─ RF-DETR her kareyi analiz eder.
    └─ En yuksek guvenilirlikli nesnenin ortalamasini alir.
    └─ 4 sn sonunda en saglam bbox ile TCTrack'i baslatir.

  TAKIP FAZI   (4 sn)
    └─ TCTrack seçili hedefi turuncu cerceve ile izler.
    └─ RF-DETR hala arka planda calismaya devam eder
       (kayip-puan duserse erken cikip yeniden tespit yapar).
    └─ 4 sn sonunda tekrar TESPIT fazina gecer.

Kullanim:
  cd /home/tom/Documents/FIREFLY
  python3 tests/python/test_rfdetr_tctrack.py

Kontroller:
  Q   : Cikis
  R   : Mevcut fazi aninda sifirla (TESPIT fazina don)

Gereksinimler:
  - pycuda, tensorrt, opencv-python-headless, mss, torch, numpy
  - TCTrack modeli : src/python/vision/TCTrack/snapshot/general_model.pth
  - TRT motoru     : object_detection/inference_model.engine
"""

from __future__ import annotations

import sys
import time
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import mss
import numpy as np

# ─── Proje kök dizinini Python yoluna ekle ───────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]          # .../FIREFLY
sys.path.insert(0, str(_ROOT))

_TCTRACK_DIR = _ROOT / "src" / "python" / "vision" / "TCTrack"
sys.path.insert(0, str(_TCTRACK_DIR))

# ─── RF-DETR modülünden ihtiyaç duyulan parçalar ────────────────────────────
from src.python.vision.rfdetr_inference.run_rfdetr_tensorrt import (
    TRTInferencer,
    ScreenCapture,
    decode_outputs,
    apply_nms,
    CONF_THRESHOLD,
    NMS_IOU_THRESHOLD,
    CLASS_NAMES,
)

# ─── TCTrack ─────────────────────────────────────────────────────────────────
import torch
from pysot.core.config import cfg
from pysot.models.utile_tctrack.model_builder import ModelBuilder_tctrack
from pysot.tracker.tctrack_tracker import TCTrackTracker
from pysot.utils.model_load import load_pretrain

# ═══════════════════════════════════════════════════════════════════════════════
#  YAPILANDIRMA
# ═══════════════════════════════════════════════════════════════════════════════
ENGINE_PATH    = str(_ROOT / "object_detection" / "inference_model.engine")
TCTRACK_CONFIG = str(_TCTRACK_DIR / "experiments" / "TCTrack" / "config.yaml")
TCTRACK_SNAP   = str(_TCTRACK_DIR / "snapshot" / "general_model.pth")

MONITOR_INDEX  = 1
RESIZE_SCALE   = 0.9
OUTPUT_FPS     = 30.0
OUTPUT_PATH    = (
    f"/home/tom/Documents/Projeler/DogFight/"
    f"rfdetr_tctrack_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
)

# Döngü süresi (saniye)
DETECT_DURATION = 1.0   # RF-DETR ile tespit fazi suresi
TRACK_DURATION  = 4.0   # TCTrack takip fazi suresi

# Tespit fazında bbox toplayıp ortalamasını almak için min kare sayısı
MIN_DET_FRAMES  = 2

# TCTrack erken cikis esigi
TRACKER_LOST_SCORE = 0.20

# ─── Renkler (BGR) ───────────────────────────────────────────────────────────
COL_DETR    = (0, 220, 90)    # yeşil  – tespit kutusu
COL_TRACK   = (0, 140, 255)   # turuncu – takip kutusu
COL_LOST    = (0, 0, 220)     # kırmızı – kayıp
COL_TEXT    = (255, 255, 255)
COL_BG      = (20, 20, 20)
COL_TIMER   = (0, 255, 220)   # timer çubuğu rengi


# ═══════════════════════════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════════════
def xywh_from_xyxy(box: np.ndarray) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def xyxy_from_tc(bbox: list) -> np.ndarray:
    """TCTrack [x,y,w,h] -> xyxy."""
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def put_text_bg(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    scale: float = 0.65,
    thickness: int = 1,
    fg: tuple = COL_TEXT,
    bg: tuple = COL_BG,
) -> None:
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
    """Ekranın üstüne yatay ilerleme çubuğu çizer."""
    h, w = frame.shape[:2]
    bar_h  = 8
    fill_w = int(w * min(elapsed / max(total, 1e-6), 1.0))
    cv2.rectangle(frame, (0, 0), (w, bar_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, 0), (fill_w, bar_h), color, -1)
    remaining = max(0.0, total - elapsed)
    put_text_bg(frame, f"{label} — kalan {remaining:.1f}s",
                (10, bar_h + 18), scale=0.6, fg=color)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    phase: str,
    n_det: int,
) -> None:
    h, w = frame.shape[:2]
    lines = [
        (f"FPS: {fps:.1f}",       (10, 55),  0.9, (0, 255, 0)),
        (f"Tespit: {n_det}",       (10, 85),  0.7, (0, 255, 255)),
        (f"Faz: {phase}",          (10, 112), 0.7, (255, 255, 0)),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for text, pos, scale, color in lines:
        cv2.putText(frame, text, pos, font, scale, color, 2, cv2.LINE_AA)

    # Yardım
    put_text_bg(frame, "Q: Cikis | R: Sifirla", (w - 200, h - 15), scale=0.5)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL YÜKLEME
# ═══════════════════════════════════════════════════════════════════════════════
def load_tctrack() -> tuple[TCTrackTracker, list]:
    cfg.merge_from_file(TCTRACK_CONFIG)
    cfg.CUDA = torch.cuda.is_available()
    model = ModelBuilder_tctrack("test")
    model = load_pretrain(model, TCTRACK_SNAP).eval()
    if cfg.CUDA:
        model = model.cuda()
    tracker = TCTrackTracker(model)
    hp = [cfg.TRACK.PENALTY_K, cfg.TRACK.WINDOW_INFLUENCE, cfg.TRACK.LR]
    return tracker, hp


# ═══════════════════════════════════════════════════════════════════════════════
#  INFERENCE YARDIMCISI
# ═══════════════════════════════════════════════════════════════════════════════
def run_detr(
    inferencer: TRTInferencer,
    frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RF-DETR ile tam frame inference + NMS. (xyxy, confs, class_ids) döner."""
    h, w = frame.shape[:2]
    outputs = inferencer.infer(frame)
    xyxy, confs, class_ids = decode_outputs(outputs, h, w, CONF_THRESHOLD)
    if len(xyxy) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )
    keep = apply_nms(xyxy, confs, class_ids, CONF_THRESHOLD * 0.5, NMS_IOU_THRESHOLD)
    if len(keep) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )
    return xyxy[keep], confs[keep], class_ids[keep]


# ═══════════════════════════════════════════════════════════════════════════════
#  ANA DÖNGÜ
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── Modeller ─────────────────────────────────────────────────────────────
    print("[INFO] RF-DETR TRT engine yukleniyor...")
    inferencer = TRTInferencer(ENGINE_PATH)
    inferencer.warmup(n=3)

    print("[INFO] TCTrack modeli yukleniyor...")
    tracker, hp = load_tctrack()
    print("[INFO] Her iki model hazir.")

    # ── Ekran yakalama ────────────────────────────────────────────────────────
    with mss.MSS() as sct:
        if MONITOR_INDEX >= len(sct.monitors):
            raise RuntimeError(f"Gecersiz monitor indexi: {MONITOR_INDEX}")
        monitor_info = sct.monitors[MONITOR_INDEX]
    print(f"[INFO] Monitor: {monitor_info}")
    print(f"[INFO] Cikti: {OUTPUT_PATH}")

    capture = ScreenCapture(MONITOR_INDEX, RESIZE_SCALE)
    print("[INFO] Ilk kare bekleniyor...")
    while capture.get() is None:
        time.sleep(0.005)
    print("[INFO] Basladi. Q=Cikis | R=Sifirla")

    # ── Pencere ───────────────────────────────────────────────────────────────
    win_name = "RF-DETR + TCTrack Auto"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # ── Video yazıcı ──────────────────────────────────────────────────────────
    video_writer: Optional[cv2.VideoWriter] = None

    # ── Durum değişkenleri ────────────────────────────────────────────────────
    prev_time  = time.perf_counter()
    fps        = 0.0
    fps_alpha  = 0.9

    # Faz zamanlayıcısı
    phase_start = time.perf_counter()  # mevcut fazın başlama anı
    in_track    = False                # False → TESPIT, True → TAKIP

    # Tespit fazında bbox toplayıcı
    # Her frame'de en yüksek güvenilirlikli bbox'ı biriktiririz.
    det_boxes:  list[np.ndarray] = []  # her frame'deki en iyi xyxy
    det_confs:  list[float]      = []
    det_cls:    list[int]        = []

    # Takip fazında son bilinen bbox
    track_xyxy: Optional[np.ndarray] = None
    track_lost  = False

    print(f"[INFO] TESPIT fazi basladi ({DETECT_DURATION:.0f}s)")

    try:
        while True:
            frame = capture.get()
            if frame is None:
                continue
            frame = frame.copy()
            h, w  = frame.shape[:2]

            # ── FPS ──────────────────────────────────────────────────────────
            now = time.perf_counter()
            dt  = now - prev_time
            if dt > 0:
                cur_fps = 1.0 / dt
                fps = fps_alpha * fps + (1.0 - fps_alpha) * cur_fps if fps > 0 else cur_fps
            prev_time = now

            # ── RF-DETR (her zaman çalışır) ──────────────────────────────────
            xyxy, confs, class_ids = run_detr(inferencer, frame)

            elapsed      = now - phase_start
            phase_done   = False

            # ═════════════════════════════════════════════════════════════════
            # TESPIT FAZI
            # ═════════════════════════════════════════════════════════════════
            if not in_track:
                phase = "TESPIT"

                # Tüm tespit kutularını yeşil çiz
                for box, conf, cid in zip(xyxy, confs, class_ids):
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COL_DETR, 2)
                    lbl = f"{CLASS_NAMES.get(int(cid), str(cid))} {conf:.2f}"
                    put_text_bg(frame, lbl, (x1, y2 + 14), scale=0.45,
                                fg=COL_DETR, bg=COL_BG)

                # En yüksek güvenilirlikli tespiti biriktir
                if len(confs) > 0:
                    best_i = int(np.argmax(confs))
                    det_boxes.append(xyxy[best_i].copy())
                    det_confs.append(float(confs[best_i]))
                    det_cls.append(int(class_ids[best_i]))

                draw_timer_bar(frame, elapsed, DETECT_DURATION, "TESPIT", COL_DETR)

                # Süre doldu → takip fazına geç
                if elapsed >= DETECT_DURATION:
                    phase_done = True
                    if len(det_boxes) >= MIN_DET_FRAMES:
                        # En yüksek conf'a sahip frame'in bbox'ını kullan
                        best_frame = int(np.argmax(det_confs))
                        init_box   = det_boxes[best_frame]
                        init_cls   = det_cls[best_frame]
                        init_xywh  = xywh_from_xyxy(init_box)

                        tracker.init(frame, init_xywh)
                        in_track    = True
                        track_lost  = False
                        track_xyxy  = init_box.copy()

                        cls_name = CLASS_NAMES.get(init_cls, str(init_cls))
                        print(
                            f"[INFO] TAKIP fazina gecildi — "
                            f"{cls_name} conf={det_confs[best_frame]:.2f} "
                            f"bbox={init_xywh}"
                        )
                    else:
                        # Yeterli tespit yoksa tespit fazını yenile
                        print(f"[WARN] Yeterli tespit toplanamadi ({len(det_boxes)} kare). "
                              f"Tespit fazi yenilendi.")
                        in_track = False

                    # Tampon temizle + zamanlayıcıyı sıfırla
                    det_boxes.clear(); det_confs.clear(); det_cls.clear()
                    phase_start = time.perf_counter()

            # ═════════════════════════════════════════════════════════════════
            # TAKIP FAZI
            # ═════════════════════════════════════════════════════════════════
            else:
                phase = "TAKIP"

                tc_out   = tracker.track(frame, hp)
                tc_score = float(tc_out["best_score"])
                tc_xyxy  = xyxy_from_tc(tc_out["bbox"])

                track_lost = tc_score < TRACKER_LOST_SCORE

                if not track_lost:
                    track_xyxy = tc_xyxy.copy()
                    x1, y1, x2, y2 = int(tc_xyxy[0]), int(tc_xyxy[1]), int(tc_xyxy[2]), int(tc_xyxy[3])
                    # Turuncu, kalın kutu
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COL_TRACK, 3)
                    # Merkez nokta
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.drawMarker(frame, (cx, cy), COL_TRACK,
                                   cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
                    put_text_bg(frame, f"TC: {tc_score:.2f}", (x1, y1 - 2),
                                scale=0.6, fg=COL_TRACK)
                else:
                    # Kayıp — son konum soluk
                    if track_xyxy is not None:
                        overlay = frame.copy()
                        x1, y1, x2, y2 = (int(v) for v in track_xyxy)
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), COL_LOST, 2)
                        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
                    put_text_bg(frame, "HEDEF KAYBOLDU",
                                (w // 2 - 90, h // 2), scale=0.8, fg=COL_LOST)

                draw_timer_bar(frame, elapsed, TRACK_DURATION, "TAKIP", COL_TRACK)

                # Süre doldu VEYA hedef tamamen kayboldu → tespit fazına dön
                if elapsed >= TRACK_DURATION or track_lost:
                    reason = "sure doldu" if elapsed >= TRACK_DURATION else "hedef kayboldu"
                    print(f"[INFO] TESPIT fazina donuluyor ({reason}).")
                    in_track    = False
                    track_xyxy  = None
                    track_lost  = False
                    det_boxes.clear(); det_confs.clear(); det_cls.clear()
                    phase_start = time.perf_counter()

            # ── HUD ──────────────────────────────────────────────────────────
            draw_hud(frame, fps, phase, len(xyxy))

            # ── Video kayıt ───────────────────────────────────────────────────
            if video_writer is None:
                oh, ow = frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    OUTPUT_PATH,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    OUTPUT_FPS,
                    (ow, oh),
                )
                if not video_writer.isOpened():
                    print(f"[WARN] Video kaydi baslatilamadi: {OUTPUT_PATH}")
                    video_writer = None

            if video_writer:
                video_writer.write(frame)

            cv2.imshow(win_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                # Mevcut fazı anında sıfırla → tespit fazına dön
                print("[INFO] Manuel sifirlama — TESPIT fazina donuluyor.")
                in_track    = False
                track_xyxy  = None
                track_lost  = False
                det_boxes.clear(); det_confs.clear(); det_cls.clear()
                phase_start = time.perf_counter()

    finally:
        capture.stop()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Kayit tamamlandi: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
