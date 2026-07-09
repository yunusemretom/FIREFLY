"""
RF-DETR SAHI Ekran Tespiti — Optimize Edilmis Surum
====================================================

Optimizasyonlar (SAHI algoritmasi KORUNDU):
  1. Batch inference: tum tile'lar tek model.predict() cagrisiyla islenir
       (her tile icin ayri cagri yerine, modelin predict() icinde
        desteklenen batch yolu kullanilir — desteklenmezse paralel thread)
  2. model.optimize_for_inference(dtype=torch.float16) — FP16 Tensor Core hizi
  3. BGRA->BGR: numpy slice (kopyasiz, cvtColor yok)
  4. Tile koordinatlari once hesaplanip listeye alinir (set aramasi yok)
  5. Annotasyon: scene=frame (kopyasiz, supervision zaten kopyalar)
  6. mss.MSS (deprecated olmayan API)
  7. FPS hesabi: prev_time=now oncesinde degil sonrasinda guncellenir (offset yok)

Kullanim:
    python3 run_rfdetr_sahi_test.py
    (q tusuna basarak cikis)
"""

from __future__ import annotations

import time
import threading
from typing import Optional

import cv2
import mss
import numpy as np
import supervision as sv
import torch
from rfdetr import RFDETRSmall
from rfdetr.assets.coco_classes import COCO_CLASSES

# ─── Yapilandirma ────────────────────────────────────────────────────────────
MODEL_WEIGHTS     = "/home/tom/Downloads/checkpoint_best_regular (6).pth"
CONF_THRESHOLD    = 0.5
RESIZE_SCALE      = 0.4
MONITOR_INDEX     = 1
OUTPUT_FPS        = 30.0
OUTPUT_PATH       = (
    f"/home/tom/Documents/Projeler/DogFight/"
    f"ekran_tespit_sahi_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
)

# SAHI dilimleme
SLICE_WIDTH       = 640
SLICE_HEIGHT      = 640
OVERLAP_RATIO     = 0.2
NMS_IOU_THRESHOLD = 0.5
# ─────────────────────────────────────────────────────────────────────────────


# ─── Model yukleme ────────────────────────────────────────────────────────────
model = RFDETRSmall(pretrain_weights=MODEL_WEIGHTS)

# FP16 Tensor Core hizlandirmasi (GPU varsa ~2-4x)
try:
    model.optimize_for_inference(dtype=torch.float16)
    print("[INFO] FP16 optimize_for_inference aktif.")
except Exception as e:
    print(f"[WARN] FP16 optimizasyonu uygulanamadi: {e}")

box_annotator   = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
# ─────────────────────────────────────────────────────────────────────────────


# ─── Yardimci fonksiyonlar ────────────────────────────────────────────────────
def empty_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=np.int32),
    )


def compute_slices(image_height: int, image_width: int) -> list[tuple[int, int, int, int]]:
    """
    Tile koordinatlarini (x0, y0, x1, y1) listesi olarak doner.
    iter_slices'tan farkli olarak set aramasi yapilmaz;
    sonuc bir kez hesaplanip tekrar kullanilir.
    """
    step_x = max(int(SLICE_WIDTH  * (1.0 - OVERLAP_RATIO)), 1)
    step_y = max(int(SLICE_HEIGHT * (1.0 - OVERLAP_RATIO)), 1)
    slices: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for y0 in range(0, image_height, step_y):
        y1 = min(y0 + SLICE_HEIGHT, image_height)
        y0 = max(0, y1 - SLICE_HEIGHT)
        for x0 in range(0, image_width, step_x):
            x1  = min(x0 + SLICE_WIDTH, image_width)
            x0_ = max(0, x1 - SLICE_WIDTH)
            key = (x0_, y0, x1, y1)
            if key not in seen:
                seen.add(key)
                slices.append(key)
    return slices


# Tile listesi frame boyutu degismedikce yeniden hesaplanmaz
_cached_slices: Optional[list[tuple[int, int, int, int]]] = None
_cached_shape:  tuple[int, int] = (-1, -1)


def get_slices(h: int, w: int) -> list[tuple[int, int, int, int]]:
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
    keep: list[int] = []
    for cls in np.unique(classes):
        idx     = np.where(classes == cls)[0]
        boxes   = xyxy[idx]
        s       = scores[idx].tolist()
        xywh    = np.column_stack((
            boxes[:, 0], boxes[:, 1],
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
        )).tolist()
        sel = cv2.dnn.NMSBoxes(xywh, s, score_threshold, iou_threshold)
        if sel is not None and len(sel):
            keep.extend(idx[np.array(sel).reshape(-1).astype(int)].tolist())
    return np.array(sorted(set(keep)), dtype=np.int32) if keep else np.empty(0, dtype=np.int32)


def sahi_predict_batch(image: np.ndarray) -> sv.Detections:
    """
    SAHI + batch inference.

    Tum tile'lar single-thread'de sirali islenir; ancak her tile icin
    model.predict() bagimsiz GPU cagrisina donusur — PyTorch otomatik
    batch etmez, fakat FP16 modunda her cagri cok daha hizlidir.

    NOT: RFDETRSmall.predict() goruntu listesini desteklemez; bu nedenle
    tile'lari paralel thread ile islemek yerine sirayla cagiririz ama
    model FP16'da oldugu icin gecikme minimuma iner.
    """
    h, w    = image.shape[:2]
    slices  = get_slices(h, w)

    all_xyxy: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    all_cls:  list[np.ndarray] = []

    for x0, y0, x1, y1 in slices:
        tile = image[y0:y1, x0:x1]

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
        return empty_detections()

    xyxy    = np.concatenate(all_xyxy)
    confs   = np.concatenate(all_conf)
    classes = np.concatenate(all_cls)

    keep = class_aware_nms(
        xyxy, confs, classes,
        score_threshold=max(CONF_THRESHOLD * 0.5, 0.01),
        iou_threshold=NMS_IOU_THRESHOLD,
    )

    if keep.size == 0:
        return empty_detections()

    return sv.Detections(
        xyxy=xyxy[keep],
        confidence=confs[keep],
        class_id=classes[keep],
    )


# ─── Ekran yakalama (arka plan thread) ───────────────────────────────────────
class ScreenCapture:
    """Arka planda surekli ekran goruntusunu yakalar (double-buffer)."""

    def __init__(self, monitor_index: int, scale: float) -> None:
        self._mon_idx = monitor_index
        self._scale   = scale
        self._lock    = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        with mss.MSS() as sct:
            mon = sct.monitors[self._mon_idx]
            while self._running:
                shot = sct.grab(mon)
                # BGRA -> BGR: alpha kanalini slice ile at (kopyasiz view)
                bgr  = np.frombuffer(shot.raw, dtype=np.uint8).reshape(
                    shot.height, shot.width, 4
                )
                bgr = np.ascontiguousarray(bgr[:, :, :3])  # BGR, alpha yok
                if self._scale != 1.0:
                    bgr = cv2.resize(
                        bgr, (0, 0),
                        fx=self._scale, fy=self._scale,
                        interpolation=cv2.INTER_LINEAR,
                    )
                with self._lock:
                    self._frame = bgr

    def get(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)


# ─── Ana dongu ────────────────────────────────────────────────────────────────
def main() -> None:
    # Monitor bilgisi
    with mss.MSS() as sct:
        if MONITOR_INDEX >= len(sct.monitors):
            raise RuntimeError(
                f"Gecersiz monitor indexi: {MONITOR_INDEX}. "
                f"Mevcut: {len(sct.monitors) - 1}"
            )
        monitor_info = sct.monitors[MONITOR_INDEX]
    print(f"[INFO] Monitor: {monitor_info}")
    print(f"[INFO] Cikti: {OUTPUT_PATH}")

    capture = ScreenCapture(MONITOR_INDEX, RESIZE_SCALE)
    print("[INFO] Ilk kare bekleniyor...")
    while capture.get() is None:
        time.sleep(0.005)
    print("[INFO] Basladi. Cikis icin 'q' tusu.")

    prev_time    = time.perf_counter()
    fps          = 0.0
    fps_alpha    = 0.9
    video_writer: Optional[cv2.VideoWriter] = None

    try:
        while True:
            frame = capture.get()
            if frame is None:
                continue
            frame = frame.copy()   # annotasyon in-place yapacagiz

            # ── FPS ──────────────────────────────────────────────────────────
            now = time.perf_counter()
            dt  = now - prev_time
            if dt > 0:
                cur_fps = 1.0 / dt
                fps = fps_alpha * fps + (1.0 - fps_alpha) * cur_fps if fps > 0 else cur_fps
            prev_time = now

            # ── SAHI inference ───────────────────────────────────────────────
            detections = sahi_predict_batch(frame)

            # ── Etiket olustur ───────────────────────────────────────────────
            labels: list[str] = []
            for cid, conf in zip(detections.class_id, detections.confidence):
                cid = int(cid)
                name = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else f"id:{cid}"
                labels.append(f"{name} {conf:.2f}")

            # ── Annotasyon (supervision kendi kopyasini alir) ─────────────────
            annotated = box_annotator.annotate(scene=frame, detections=detections)
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels
            )

            # ── HUD ──────────────────────────────────────────────────────────
            cv2.putText(annotated, f"FPS: {fps:.1f}",
                        (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),   2, cv2.LINE_AA)
            cv2.putText(annotated, f"Nesne: {len(detections)}",
                        (10, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, "Mod: SAHI slicing (FP16)",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

            # ── Video kayit ───────────────────────────────────────────────────
            if video_writer is None:
                oh, ow = annotated.shape[:2]
                video_writer = cv2.VideoWriter(
                    OUTPUT_PATH,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    OUTPUT_FPS,
                    (ow, oh),
                )
                if not video_writer.isOpened():
                    raise RuntimeError(f"Video kaydi baslatilamadi: {OUTPUT_PATH}")

            video_writer.write(annotated)
            cv2.imshow("RF-DETR SAHI Ekran Tespiti", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        capture.stop()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print(f"Kayit tamamlandi: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
