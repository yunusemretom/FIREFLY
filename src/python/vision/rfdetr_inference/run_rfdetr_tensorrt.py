"""
RF-DETR TensorRT Inference - Ekran Yakalama + SAHI (Optimize)
TensorRT 11 / Python 3.10 uyumlu

Mevcut optimizasyonlar KORUNDU, SAHI yeniden eklendi:
  - SAHI dilimleme: kucuk nesneleri yakalamak icin tile bazli inference
  - Tile koordinat listesi once hesaplanip onbelleklendi (her frame'de set aramasi yok)
  - Preprocessing buffer'i once tahsis edildi (her tile icin malloc yok)
  - mean/std CHW formatinda sabit (modül seviyesi, bir kez)
  - BGRA->BGR numpy slice (cvtColor yok)
  - Double-buffer + threading: ekran yakalama ve inference paralel
  - Annotasyon in-place (frame.copy() yok)
  - Warmup inference: soguk baslatma gecikmesini atar
  - Pinned memory + async memcpy

Kullanim:
    python run_rfdetr_tensorrt.py
    (q tusuna basarak cikis)
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import mss
import numpy as np

# ─── Yapilandirma ────────────────────────────────────────────────────────────
ENGINE_PATH    = "object_detection/inference_model.engine"
CONF_THRESHOLD = 0.45
RESIZE_SCALE   = 0.4          # ekran boyutu carpani
MONITOR_INDEX  = 1            # 1 = ana ekran
OUTPUT_FPS     = 50.0
OUTPUT_PATH    = (
    f"/home/tom/Documents/Projeler/DogFight/"
    f"ekran_tespit_trt_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
)

# SAHI dilimleme parametreleri
SLICE_WIDTH    = 640
SLICE_HEIGHT   = 640
OVERLAP_RATIO  = 0.2
NMS_IOU_THRESHOLD = 0.45

# Sinif tanimlari
CLASS_NAMES = {
    0: "drone",
    1: "f16",
    2: "helicopter",
    3: "rocket",
    4: "missile",
}

# Disa aktarilan sabit
EXCLUSION_ZONES: list[tuple[int, int, int, int]] = []  # [(x1,y1,x2,y2), ...]

# Annotasyon renkleri (BGR)
BOX_COLOR   = (0, 220, 90)
LABEL_COLOR = (255, 255, 255)
LABEL_BG    = (0, 150, 60)
# ─────────────────────────────────────────────────────────────────────────────


# ─── TensorRT / CUDA Yukleme ─────────────────────────────────────────────────
try:
    import pycuda.driver as cuda_drv
    import pycuda.autoinit  # noqa: F401
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False

if not HAS_PYCUDA:
    raise ImportError("pycuda bulunamadi!\nKur: pip install pycuda")

try:
    import tensorrt as trt
except ImportError:
    raise ImportError(
        "TensorRT Python modulu bulunamadi!\n"
        "Sisteme kuruluysa: export PYTHONPATH=/usr/lib/python3/dist-packages\n"
        "Ya da: pip install tensorrt-cu12 --extra-index-url https://pypi.nvidia.com"
    )
# ─────────────────────────────────────────────────────────────────────────────


# ─── Preprocess sabitleri (modül seviyesinde, bir kez) ───────────────────────
_MEAN   = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD    = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
_INV255 = np.float32(1.0 / 255.0)
# ─────────────────────────────────────────────────────────────────────────────


class TRTInferencer:
    """
    TensorRT engine ile FP16/FP32 inference.

    Optimizasyonlar:
      - Pinned input buffer once tahsis edilir.
      - mean/std CHW buffer'ina dogrudan uygulanir.
      - Tensor adresleri init'te bir kez baglanir.
      - warmup() ile soguk baslatma atlatilir.
    """

    def __init__(self, engine_path: str):
        path = Path(engine_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Engine dosyasi bulunamadi: {path.resolve()}\n"
                "Once export_rfdetr_tensorrt.py ile export yapin."
            )

        self.logger  = trt.Logger(trt.Logger.WARNING)
        runtime      = trt.Runtime(self.logger)

        print(f"[INFO] TRT engine yukleniyor: {path.resolve()}")
        with open(path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Engine deserialize edilemedi!")

        self.context = self.engine.create_execution_context()

        self.input_names:  list[str] = []
        self.output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        print(f"[INFO] Inputs : {self.input_names}")
        print(f"[INFO] Outputs: {self.output_names}")

        self._init_pycuda()
        print("[INFO] TRT Inferencer hazir.")

    def _init_pycuda(self) -> None:
        self._host_inputs:  dict[str, np.ndarray]              = {}
        self._host_outputs: dict[str, np.ndarray]              = {}
        self._cuda_inputs:  dict[str, cuda_drv.DeviceAllocation] = {}
        self._cuda_outputs: dict[str, cuda_drv.DeviceAllocation] = {}

        in_name           = self.input_names[0]
        shape             = tuple(self.engine.get_tensor_shape(in_name))
        self._input_shape: tuple[int, ...] = shape
        _, _, H, W        = shape
        size              = int(np.prod(shape))

        pinned_in = cuda_drv.pagelocked_empty(size, dtype=np.float32)
        self._host_inputs[in_name] = pinned_in
        self._cuda_inputs[in_name] = cuda_drv.mem_alloc(pinned_in.nbytes)

        # CHW view — preprocess dogrudan buraya yazar
        self._chw_buf: np.ndarray  = pinned_in.reshape(3, H, W)
        # Resize tamponu — her infer() de yeniden ayrilmaz
        self._resize_buf: np.ndarray = np.empty((H, W, 3), dtype=np.uint8)

        for out_name in self.output_names:
            out_shape = tuple(self.engine.get_tensor_shape(out_name))
            size      = int(np.prod(np.abs(out_shape)))
            host      = cuda_drv.pagelocked_empty(size, dtype=np.float32)
            self._host_outputs[out_name] = host
            self._cuda_outputs[out_name] = cuda_drv.mem_alloc(host.nbytes)
            setattr(self, f"_shape_{out_name}", out_shape)

        self._stream = cuda_drv.Stream()

        # Tensor adreslerini bir kez bagla
        self.context.set_tensor_address(in_name, int(self._cuda_inputs[in_name]))
        for out_name in self.output_names:
            self.context.set_tensor_address(out_name, int(self._cuda_outputs[out_name]))

    def warmup(self, n: int = 3) -> None:
        """GPU'yu isit — TRT soguk baslatma JIT gecikmesini atar."""
        _, _, H, W = self._input_shape
        dummy = np.zeros((H, W, 3), dtype=np.uint8)
        for _ in range(n):
            self.infer(dummy)
        print(f"[INFO] Warmup tamamlandi ({n} gecis).")

    def _preprocess_into_pinned(self, image: np.ndarray) -> None:
        """BGR uint8 -> pinned CHW float32 (normalizasyon dahil, sifir kopya)."""
        _, _, H, W = self._input_shape
        cv2.resize(image, (W, H), dst=self._resize_buf)
        rgb = self._resize_buf[:, :, ::-1]   # BGR->RGB (view, kopyasiz)
        chw = rgb.transpose(2, 0, 1).astype(np.float32, copy=False)
        chw *= _INV255
        chw -= _MEAN
        chw /= _STD
        np.copyto(self._chw_buf, chw)

    def infer(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Verilen frame uzerinde inference yap. dict{output_name: array} doner."""
        self._preprocess_into_pinned(image)

        in_name = self.input_names[0]
        cuda_drv.memcpy_htod_async(self._cuda_inputs[in_name], self._host_inputs[in_name], self._stream)
        self.context.execute_async_v3(self._stream.handle)
        for out_name in self.output_names:
            cuda_drv.memcpy_dtoh_async(self._host_outputs[out_name], self._cuda_outputs[out_name], self._stream)
        self._stream.synchronize()

        results: dict[str, np.ndarray] = {}
        for out_name in self.output_names:
            shape = getattr(self, f"_shape_{out_name}")
            results[out_name] = self._host_outputs[out_name].reshape(shape).copy()
        return results

    def __del__(self) -> None:
        try:
            if hasattr(self, "_stream"):
                self._stream.synchronize()
        except Exception:
            pass


# ─── Cikti decode ────────────────────────────────────────────────────────────
def decode_outputs(
    outputs: dict[str, np.ndarray],
    orig_h: int,
    orig_w: int,
    conf_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RF-DETR ciktilarini (dets, labels) decode eder."""
    dets_key = next(
        (k for k in outputs if "det" in k.lower() or "box" in k.lower()),
        list(outputs.keys())[0],
    )
    labels_key = next(
        (k for k in outputs if "label" in k.lower() or "class" in k.lower() or "logit" in k.lower()),
        list(outputs.keys())[1],
    )

    dets   = outputs[dets_key][0]    # (300, 4)
    logits = outputs[labels_key][0]  # (300, num_classes)

    scores    = 1.0 / (1.0 + np.exp(-logits))   # sigmoid
    class_ids = np.argmax(scores, axis=-1)
    confs     = scores[np.arange(len(class_ids)), class_ids]

    mask = confs >= conf_threshold
    if not mask.any():
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )

    dets      = dets[mask]
    class_ids = class_ids[mask].astype(np.int32)
    confs     = confs[mask].astype(np.float32)

    cx, cy, w, h = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    x1 = np.clip((cx - w * 0.5) * orig_w, 0, orig_w)
    y1 = np.clip((cy - h * 0.5) * orig_h, 0, orig_h)
    x2 = np.clip((cx + w * 0.5) * orig_w, 0, orig_w)
    y2 = np.clip((cy + h * 0.5) * orig_h, 0, orig_h)

    xyxy = np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)
    return xyxy, confs, class_ids


def apply_nms(
    xyxy: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    score_threshold: float,
    iou_threshold: float,
) -> np.ndarray:
    """Sinif-bazli NMS. Gecerli indis dizisi doner."""
    if len(xyxy) == 0:
        return np.empty(0, dtype=np.int32)
    keep: list[int] = []
    for cls in np.unique(classes):
        idx  = np.where(classes == cls)[0]
        boxes = xyxy[idx]
        s    = scores[idx].tolist()
        xywh = np.column_stack((
            boxes[:, 0], boxes[:, 1],
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
        )).tolist()
        sel = cv2.dnn.NMSBoxes(xywh, s, score_threshold, iou_threshold)
        if sel is not None and len(sel):
            keep.extend(idx[np.array(sel).reshape(-1)].tolist())
    return np.array(sorted(set(keep)), dtype=np.int32) if keep else np.empty(0, dtype=np.int32)


# ─── SAHI ────────────────────────────────────────────────────────────────────
# Tile koordinat listesi: frame boyutu degismedikce yeniden hesaplanmaz
_sahi_cache_shape: tuple[int, int] = (-1, -1)
_sahi_slices:      list[tuple[int, int, int, int]] = []


def _compute_slices(img_h: int, img_w: int) -> list[tuple[int, int, int, int]]:
    """Tile koordinatlarini (x0,y0,x1,y1) listesi olarak hesaplar."""
    step_x = max(int(SLICE_WIDTH  * (1.0 - OVERLAP_RATIO)), 1)
    step_y = max(int(SLICE_HEIGHT * (1.0 - OVERLAP_RATIO)), 1)
    slices: list[tuple[int, int, int, int]] = []
    seen:   set[tuple[int, int, int, int]]  = set()
    for y0 in range(0, img_h, step_y):
        y1  = min(y0 + SLICE_HEIGHT, img_h)
        y0_ = max(0, y1 - SLICE_HEIGHT)
        for x0 in range(0, img_w, step_x):
            x1  = min(x0 + SLICE_WIDTH, img_w)
            x0_ = max(0, x1 - SLICE_WIDTH)
            key = (x0_, y0_, x1, y1)
            if key not in seen:
                seen.add(key)
                slices.append(key)
    return slices


def _get_slices(h: int, w: int) -> list[tuple[int, int, int, int]]:
    global _sahi_cache_shape, _sahi_slices
    if (h, w) != _sahi_cache_shape:
        _sahi_slices      = _compute_slices(h, w)
        _sahi_cache_shape = (h, w)
    return _sahi_slices


def sahi_predict(
    inferencer: TRTInferencer,
    frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SAHI dilimleme ile inference.
    Her tile icin ayri TRT cagrisi yapilir; sonuclar birlestirilip NMS uygulanir.
    Tile koordinat listesi onbelleklenir — frame boyutu degismedikce yeniden hesaplanmaz.
    """
    h, w   = frame.shape[:2]
    slices = _get_slices(h, w)

    all_xyxy: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    all_cls:  list[np.ndarray] = []

    for x0, y0, x1, y1 in slices:
        tile = frame[y0:y1, x0:x1]   # view (kopyasiz)
        outputs = inferencer.infer(tile)
        tile_h, tile_w = tile.shape[:2]
        xyxy, confs, class_ids = decode_outputs(outputs, tile_h, tile_w, CONF_THRESHOLD)

        if len(xyxy) == 0:
            continue

        # Tile koordinatlarini ana frame koordinatlarina donustur
        xyxy[:, [0, 2]] += x0
        xyxy[:, [1, 3]] += y0
        all_xyxy.append(xyxy)
        all_conf.append(confs)
        all_cls.append(class_ids)

    if not all_xyxy:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )

    xyxy      = np.concatenate(all_xyxy)
    confs     = np.concatenate(all_conf)
    class_ids = np.concatenate(all_cls)

    keep = apply_nms(xyxy, confs, class_ids,
                     max(CONF_THRESHOLD * 0.5, 0.01), NMS_IOU_THRESHOLD)
    if len(keep) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )

    return xyxy[keep], confs[keep], class_ids[keep]


# ─── Ekran yakalama (arka plan thread) ───────────────────────────────────────
class ScreenCapture:
    """
    Arka planda surekli ekran goruntusunu yakalar (double-buffer).
    Ana dongu inference yaparken yakalama devam eder.
    """

    def __init__(self, monitor_index: int, resize_scale: float):
        self._monitor_index = monitor_index
        self._scale         = resize_scale
        self._lock          = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._running       = True
        self._thread        = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        with mss.MSS() as sct:
            mon = sct.monitors[self._monitor_index]
            while self._running:
                shot = sct.grab(mon)
                bgr  = np.frombuffer(shot.raw, dtype=np.uint8).reshape(shot.height, shot.width, 4)
                bgr  = np.ascontiguousarray(bgr[:, :, :3])   # BGRA->BGR (alpha at)
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


# ─── Annotasyon (in-place) ───────────────────────────────────────────────────
def annotate_inplace(frame: np.ndarray, xyxy: np.ndarray, confs: np.ndarray, class_ids: np.ndarray) -> None:
    """Tespitleri in-place cizer (kopya yapmaz)."""
    for box, conf, cid in zip(xyxy, confs, class_ids):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        label = f"{CLASS_NAMES.get(int(cid), f'id:{cid}')} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), LABEL_BG, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, LABEL_COLOR, 1, cv2.LINE_AA)


def put_stats(frame: np.ndarray, fps: float, n_det: int, n_tiles: int) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}",         (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),   2, cv2.LINE_AA)
    cv2.putText(frame, f"Nesne: {n_det}",          (10, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Mod: TRT SAHI ({n_tiles} tile)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)


# ─── Ana Dongu ───────────────────────────────────────────────────────────────
def main() -> None:
    inferencer = TRTInferencer(ENGINE_PATH)

    with mss.MSS() as sct:
        if MONITOR_INDEX >= len(sct.monitors):
            raise RuntimeError(
                f"Gecersiz monitor indexi: {MONITOR_INDEX}. "
                f"Mevcut: {len(sct.monitors) - 1}"
            )
        monitor = sct.monitors[MONITOR_INDEX]
    print(f"[INFO] Monitor: {monitor}")
    print(f"[INFO] Cikti: {OUTPUT_PATH}")

    inferencer.warmup(n=3)

    capture = ScreenCapture(MONITOR_INDEX, RESIZE_SCALE)
    print("[INFO] Ilk kare bekleniyor...")
    while capture.get() is None:
        time.sleep(0.005)
    print("[INFO] Cikis icin pencerede 'q' tusuna basin.")

    prev_time    = time.perf_counter()
    fps          = 0.0
    fps_alpha    = 0.9
    video_writer: Optional[cv2.VideoWriter] = None

    try:
        while True:
            frame = capture.get()
            if frame is None:
                continue
            frame = frame.copy()

            now = time.perf_counter()
            dt  = now - prev_time
            if dt > 0:
                cur_fps = 1.0 / dt
                fps = fps_alpha * fps + (1.0 - fps_alpha) * cur_fps if fps > 0 else cur_fps
            prev_time = now

            # SAHI inference
            xyxy, confs, class_ids = sahi_predict(inferencer, frame)

            # Onbellekteki tile sayisini goster
            h, w = frame.shape[:2]
            n_tiles = len(_get_slices(h, w))

            annotate_inplace(frame, xyxy, confs, class_ids)
            put_stats(frame, fps, len(xyxy), n_tiles)

            if video_writer is None:
                oh, ow = frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    OUTPUT_PATH,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    OUTPUT_FPS,
                    (ow, oh),
                )
                if not video_writer.isOpened():
                    raise RuntimeError(f"Video kaydi baslatilamadi: {OUTPUT_PATH}")

            video_writer.write(frame)
            cv2.imshow("RF-DETR TensorRT SAHI", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        capture.stop()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Kayit tamamlandi: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
