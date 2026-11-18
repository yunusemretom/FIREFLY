"""
Talon dedektoru — RF-DETR ONNX (object_detection/inference_model.onnx).

Model 512x512 giris alir. Tam kareyi 512'ye kucultmek uzak hedefi ~10 px'e
indirir; bunun yerine kestiricinin ongordugu piksel etrafindan TAM COZUNURLUKTE
512x512 bir pencere (ROI) kesilir. 8 m'de Talon ~95x30 px, 20 m'de ~35 px.

Cikis: dets [1,300,4] (cx,cy,w,h normalize), labels [1,300,3] logit.
Simulatorde Talon = sinif 1 (olculdu: skor ~0.75, konum hatasi ~5 px).
"""
import os

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_MODEL = os.path.join(ROOT, "object_detection", "inference_model.onnx")

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

# Surucu 550 = CUDA 12.4 -> CUDA 12 kutuphaneleri gerekir. ~/.local altindaki
# nvidia/cudnn klasorune CUDA 13 surumu yazilmis (calismaz); bu yuzden proje
# .venv'indeki nvidia-cudnn-cu12 ONCE aranir. Kutuphaneler mutlak yoldan
# RTLD_GLOBAL ile yuklenir; LD_LIBRARY_PATH gerekmez.
_CUDA_LIBS = [
    ("cuda_runtime", "libcudart.so.12"), ("cublas", "libcublasLt.so.12"),
    ("cublas", "libcublas.so.12"), ("cufft", "libcufft.so.11"), ("curand", "libcurand.so.10"),
    ("cudnn", "libcudnn_graph.so.9"), ("cudnn", "libcudnn_ops.so.9"),
    ("cudnn", "libcudnn_heuristic.so.9"), ("cudnn", "libcudnn_engines_runtime_compiled.so.9"),
    ("cudnn", "libcudnn_engines_precompiled.so.9"), ("cudnn", "libcudnn_cnn.so.9"),
    ("cudnn", "libcudnn_adv.so.9"), ("cudnn", "libcudnn.so.9"),
]


def _preload_cuda_libs():
    import ctypes
    import site
    import sys
    roots = [os.path.join(p, "nvidia") for p in site.getsitepackages()] if hasattr(site, "getsitepackages") else []
    roots += [os.path.join(site.getusersitepackages(), "nvidia")]
    roots += [os.path.join(sys.prefix, "lib", "python%d.%d" % sys.version_info[:2], "site-packages", "nvidia")]
    loaded = []
    for pkg, lib in _CUDA_LIBS:
        for r in roots:
            p = os.path.join(r, pkg, "lib", lib)
            if os.path.exists(p):
                try:
                    ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(p)
                    break
                except OSError:
                    continue
    return loaded


class TalonDetector:
    def __init__(self, model_path=DEFAULT_MODEL, talon_class=1, input_size=512, threads=4,
                 use_gpu=True):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = None
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            _preload_cuda_libs()
            try:
                self.sess = ort.InferenceSession(model_path, sess_options=so,
                                                 providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            except Exception as e:   # noqa: BLE001
                print(f"[TalonDetector] GPU acilamadi, CPU'ya dusuluyor: {str(e)[:120]}")
        if self.sess is None:
            self.sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.provider = self.sess.get_providers()[0]
        self.in_name = self.sess.get_inputs()[0].name
        self.cls = talon_class
        self.S = input_size

    def detect(self, frame, center=None, min_score=0.35, zoom=1.0):
        """frame: BGR. center: (u, v) ROI merkezi (None -> tum kare kucultulur).
        zoom > 1: ROI (S/zoom) kesilip S'ye buyutulur -> uzak (kucuk) hedef icin.

        Donus: [(u, v, w, h, skor), ...] tam kare pikselinde, skora gore azalan.
        """
        H, W = frame.shape[:2]
        S = self.S
        if center is not None and W >= S and H >= S:
            C = int(round(S / max(zoom, 1.0)))
            cx = int(np.clip(center[0], C // 2, W - C // 2))
            cy = int(np.clip(center[1], C // 2, H - C // 2))
            x0, y0 = cx - C // 2, cy - C // 2
            img = frame[y0:y0 + C, x0:x0 + C]
            if C != S:
                img = cv2.resize(img, (S, S), interpolation=cv2.INTER_LINEAR)
            sx = sy = C / S
        else:
            x0 = y0 = 0
            img = cv2.resize(frame, (S, S))
            sx, sy = W / S, H / S
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None]
        dets, logits = self.sess.run(None, {self.in_name: x})
        score = 1.0 / (1.0 + np.exp(-logits[0, :, self.cls]))
        idx = np.where(score >= min_score)[0]
        out = []
        for k in idx[np.argsort(-score[idx])][:5]:
            cx_, cy_, w_, h_ = dets[0, k]
            out.append((x0 + cx_ * S * sx, y0 + cy_ * S * sy, w_ * S * sx, h_ * S * sy, float(score[k])))
        return out
