"""
RF-DETR TensorRT Inference - Ekran Yakalama + SAHI Dilimleme
TensorRT 11 / Python 10 uyumlu

Gereksinimler:
    pip install tensorrt cuda-python opencv-python-headless mss numpy
    (tensorrt: sistem kurulumunu .venv icine baglayabilirsin - bkz. asagidaki not)

Not: TRT Python modulu .venv icinde yoksa, sisteme kurulu olanı kullanmak icin:
    export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH
    ya da: pip install tensorrt-cu12 --extra-index-url https://pypi.nvidia.com

Kullanim:
    python run_rfdetr_tensorrt.py
    (q tusuna basarak cikis)
"""

import time
from pathlib import Path

import cv2
import mss
import numpy as np

# ─── Yapilandirma ────────────────────────────────────────────────────────────
ENGINE_PATH    = "object_detection/inference_model.engine"
CONF_THRESHOLD = 0.45
RESIZE_SCALE   = 0.8          # ekran boyutu carpani (performans/kalite dengesi)
MONITOR_INDEX  = 1            # 1 = ana ekran
OUTPUT_FPS     = 50.0
OUTPUT_PATH    = f"/home/tom/Documents/Projeler/DogFight/ekran_tespit_trt_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

# SAHI dilimleme
SLICE_WIDTH    = 640
SLICE_HEIGHT   = 640
OVERLAP_RATIO  = 0.2
NMS_IOU_THRESHOLD = 0.45

# RF-DETR model input boyutu (export sirasinda kullanilan deger)
MODEL_INPUT_SIZE = 560  # RFDETRSmall default: 560x560

# Sinif tanimlari (egitim sirasindaki siniflar)
CLASS_NAMES = {
    0: "drone",
    1: "f16",
    2: "helicopter",
    3: "rocket",
    4: "missile",
}

# Annotasyon renkleri (BGR)
BOX_COLOR   = (0, 220, 90)
LABEL_COLOR = (255, 255, 255)
LABEL_BG    = (0, 150, 60)
# ─────────────────────────────────────────────────────────────────────────────


# ─── TensorRT Yukleme ────────────────────────────────────────────────────────
try:
    import tensorrt as trt
    import cuda
    from cuda import cudart
    HAS_CUDA_PYTHON = True
except ImportError:
    HAS_CUDA_PYTHON = False

try:
    import pycuda.driver as cuda_drv
    import pycuda.autoinit  # noqa: F401
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False

if not (HAS_CUDA_PYTHON or HAS_PYCUDA):
    raise ImportError(
        "CUDA Python binding bulunamadi!\n"
        "Kur: pip install pycuda   VEYA   pip install cuda-python"
    )

try:
    import tensorrt as trt
except ImportError:
    raise ImportError(
        "TensorRT Python modulu bulunamadi!\n"
        "Sisteme kuruluysa: export PYTHONPATH=/usr/lib/python3/dist-packages\n"
        "Ya da: pip install tensorrt-cu12 --extra-index-url https://pypi.nvidia.com"
    )
# ─────────────────────────────────────────────────────────────────────────────


class TRTInferencer:
    """TensorRT engine ile FP16/FP32 inference."""

    def __init__(self, engine_path: str):
        path = Path(engine_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Engine dosyasi bulunamadi: {path.resolve()}\n"
                "Once export_rfdetr_tensorrt.py ile export yapin."
            )

        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)

        print(f"[INFO] TRT engine yukleniyor: {path.resolve()}")
        with open(path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Engine deserialize edilemedi!")

        self.context = self.engine.create_execution_context()

        # Tensor isimlerini al
        self.input_names  = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        print(f"[INFO] Inputs : {self.input_names}")
        print(f"[INFO] Outputs: {self.output_names}")

        # CUDA bellegi ayir (pycuda yolu)
        if HAS_PYCUDA:
            self._init_pycuda()
        else:
            self._init_cuda_python()

        print("[INFO] TRT Inferencer hazir.")

    # ── pycuda yolu ──────────────────────────────────────────────────────────
    def _init_pycuda(self):
        self._use_pycuda = True
        self._host_inputs  = {}
        self._host_outputs = {}
        self._cuda_inputs   = {}
        self._cuda_outputs  = {}

        # Input: [1, 3, H, W]
        in_name = self.input_names[0]
        shape = tuple(self.engine.get_tensor_shape(in_name))  # (1, 3, 560, 560)
        size  = int(np.prod(shape))
        self._input_shape = shape
        self._host_inputs[in_name]  = cuda_drv.pagelocked_empty(size, dtype=np.float32)
        self._cuda_inputs[in_name]  = cuda_drv.mem_alloc(self._host_inputs[in_name].nbytes)

        # Outputs
        for out_name in self.output_names:
            shape = tuple(self.engine.get_tensor_shape(out_name))
            size  = int(np.prod(np.abs(shape)))  # dinamik boyutlara karsi abs
            host  = cuda_drv.pagelocked_empty(size, dtype=np.float32)
            self._host_outputs[out_name] = host
            self._cuda_outputs[out_name] = cuda_drv.mem_alloc(host.nbytes)
            setattr(self, f"_shape_{out_name}", shape)

        self._stream = cuda_drv.Stream()

    # ── cuda-python yolu ─────────────────────────────────────────────────────
    def _init_cuda_python(self):
        self._use_pycuda = False
        raise NotImplementedError("cuda-python yolu henuz desteklenmiyor, pycuda kurun.")

    # ── Onislem ──────────────────────────────────────────────────────────────
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """BGR image -> float32 CHW [0,1] normalized, model input boyutuna boyutlandir."""
        _, _, H, W = self._input_shape
        img = cv2.resize(image, (W, H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # ImageNet normalizasyon
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std
        img  = img.transpose(2, 0, 1)          # HWC -> CHW
        img  = np.ascontiguousarray(img[None])  # batch dim ekle
        return img

    # ── Inference ────────────────────────────────────────────────────────────
    def infer(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Tek bir tile/frame uzerinde inference yap. dict{output_name: array} doner."""
        inp = self._preprocess(image)

        if self._use_pycuda:
            return self._infer_pycuda(inp)

    def _infer_pycuda(self, inp: np.ndarray) -> dict[str, np.ndarray]:
        in_name = self.input_names[0]
        np.copyto(self._host_inputs[in_name], inp.ravel())

        # Host -> Device
        cuda_drv.memcpy_htod_async(self._cuda_inputs[in_name], self._host_inputs[in_name], self._stream)

        # Tensor adreslerini bağla
        self.context.set_tensor_address(in_name, int(self._cuda_inputs[in_name]))
        for out_name in self.output_names:
            self.context.set_tensor_address(out_name, int(self._cuda_outputs[out_name]))

        self.context.execute_async_v3(self._stream.handle)

        # Device -> Host
        results = {}
        for out_name in self.output_names:
            cuda_drv.memcpy_dtoh_async(self._host_outputs[out_name], self._cuda_outputs[out_name], self._stream)
        self._stream.synchronize()

        for out_name in self.output_names:
            shape = getattr(self, f"_shape_{out_name}")
            results[out_name] = self._host_outputs[out_name].reshape(shape).copy()

        return results

    def __del__(self):
        try:
            if hasattr(self, "_stream"):
                self._stream.synchronize()
        except Exception:
            pass


# ─── Sonuc Islemleri ─────────────────────────────────────────────────────────
def decode_outputs(outputs: dict, orig_h: int, orig_w: int, conf_threshold: float):
    """
    RF-DETR ONNX ciktilarini decode et.
    Cikti tensörleri:
        dets   : (1, 300, 4) - cxcywh formatinda, normalize edilmis [0,1]
        labels : (1, 300, num_classes) - logit skorlari
    """
    # Cikti isimlerini bul (dets / labels)
    dets_key   = next((k for k in outputs if "det" in k.lower() or "box" in k.lower()), list(outputs.keys())[0])
    labels_key = next((k for k in outputs if "label" in k.lower() or "class" in k.lower() or "logit" in k.lower()), list(outputs.keys())[1])

    dets   = outputs[dets_key][0]    # (300, 4)
    logits = outputs[labels_key][0]  # (300, num_classes)

    scores    = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    class_ids = np.argmax(scores, axis=-1)       # (300,)
    confs     = scores[np.arange(len(class_ids)), class_ids]  # (300,)

    # Filtrele
    mask  = confs >= conf_threshold
    dets  = dets[mask]
    class_ids = class_ids[mask]
    confs = confs[mask]

    if len(dets) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    # cxcywh -> xyxy, piksel koordinatlarina cevir
    cx, cy, w, h = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h

    xyxy = np.stack([x1, y1, x2, y2], axis=-1)
    return xyxy, confs, class_ids


def class_aware_nms(xyxy, scores, classes, score_threshold, iou_threshold):
    keep = []
    for cls in np.unique(classes):
        idx = np.where(classes == cls)[0]
        boxes  = xyxy[idx]
        s      = scores[idx].tolist()
        xywh   = np.column_stack((boxes[:, 0], boxes[:, 1],
                                   boxes[:, 2] - boxes[:, 0],
                                   boxes[:, 3] - boxes[:, 1])).tolist()
        sel = cv2.dnn.NMSBoxes(xywh, s, score_threshold, iou_threshold)
        if sel is not None and len(sel):
            keep.extend(idx[np.array(sel).reshape(-1)].tolist())
    return np.array(sorted(set(keep)), dtype=np.int32) if keep else np.empty(0, dtype=np.int32)


def iter_slices(img_h, img_w):
    step_x = max(int(SLICE_WIDTH  * (1.0 - OVERLAP_RATIO)), 1)
    step_y = max(int(SLICE_HEIGHT * (1.0 - OVERLAP_RATIO)), 1)
    seen = set()
    for y0 in range(0, img_h, step_y):
        y1 = min(y0 + SLICE_HEIGHT, img_h)
        y0 = max(0, y1 - SLICE_HEIGHT)
        for x0 in range(0, img_w, step_x):
            x1 = min(x0 + SLICE_WIDTH, img_w)
            x0 = max(0, x1 - SLICE_WIDTH)
            key = (x0, y0, x1, y1)
            if key in seen:
                continue
            seen.add(key)
            yield key


def sahi_predict(inferencer: TRTInferencer, frame: np.ndarray):
    h, w = frame.shape[:2]
    all_xyxy, all_conf, all_cls = [], [], []

    for x0, y0, x1, y1 in iter_slices(h, w):
        tile = frame[y0:y1, x0:x1]
        tile_h, tile_w = tile.shape[:2]

        outputs = inferencer.infer(tile)
        xyxy, conf, cls = decode_outputs(outputs, tile_h, tile_w, CONF_THRESHOLD)

        if len(xyxy) == 0:
            continue

        # Koordinatlari ana kareye cevir
        xyxy[:, [0, 2]] += x0
        xyxy[:, [1, 3]] += y0
        all_xyxy.append(xyxy)
        all_conf.append(conf)
        all_cls.append(cls)

    if not all_xyxy:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    xyxy  = np.concatenate(all_xyxy)
    confs = np.concatenate(all_conf)
    clses = np.concatenate(all_cls)

    keep = class_aware_nms(xyxy, confs, clses,
                           max(CONF_THRESHOLD * 0.5, 0.01),
                           NMS_IOU_THRESHOLD)
    if len(keep) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    return xyxy[keep], confs[keep], clses[keep]


# ─── Annotasyon ──────────────────────────────────────────────────────────────
def annotate(frame, xyxy, confs, class_ids):
    out = frame.copy()
    for box, conf, cid in zip(xyxy, confs, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{CLASS_NAMES.get(int(cid), f'id:{cid}')} {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), LABEL_BG, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, LABEL_COLOR, 1, cv2.LINE_AA)
    return out


def put_stats(frame, fps, n_det, mode="TRT SAHI"):
    cv2.putText(frame, f"FPS: {fps:.1f}",       (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),   2, cv2.LINE_AA)
    cv2.putText(frame, f"Nesne: {n_det}",        (10, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Mod: {mode}",           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    return frame


# ─── Ana Dongu ───────────────────────────────────────────────────────────────
def main():
    inferencer = TRTInferencer(ENGINE_PATH)

    prev_time   = time.perf_counter()
    fps         = 0.0
    fps_alpha   = 0.9
    video_writer = None

    with mss.MSS() as sct:
        if MONITOR_INDEX >= len(sct.monitors):
            raise RuntimeError(
                f"Gecersiz monitor indexi: {MONITOR_INDEX}. "
                f"Mevcut: {len(sct.monitors) - 1}"
            )

        monitor = sct.monitors[MONITOR_INDEX]
        print(f"[INFO] Monitor: {monitor}")
        print(f"[INFO] Cikti: {OUTPUT_PATH}")
        print("[INFO] Cikis icin pencerede 'q' tusuna basin.")

        while True:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            if RESIZE_SCALE != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)

            now = time.perf_counter()
            dt  = now - prev_time
            if dt > 0:
                cur_fps = 1.0 / dt
                fps = (fps_alpha * fps + (1.0 - fps_alpha) * cur_fps) if fps > 0 else cur_fps
            prev_time = now

            xyxy, confs, class_ids = sahi_predict(inferencer, frame)

            annotated = annotate(frame, xyxy, confs, class_ids)
            annotated = put_stats(annotated, fps, len(xyxy))

            # Video yazar
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
            cv2.imshow("RF-DETR TensorRT SAHI", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Kayit tamamlandi: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
