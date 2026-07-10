"""
Visual Target Acquisition (RF-DETR + TCTrack) & Route Intercept System (Telemetry / Sim)
========================================================================================

Çalıştırma:
    python3 tests/python/test_combined_tracker_intercept.py [--sim] [--headless-intercept]

Bu dosya iki ayrı sistemi tek bir ana döngüde birleştirir:
  1. Hedef Ekranı (OpenCV): Ekran yakalama + RF-DETR TRT + TCTrack ile görsel hedef izleme.
  2. Rota Ekranı (Matplotlib): drone_sdk veya Ellipse Simülasyonu ile rotayı çözme ve dron hareketi.
"""

from __future__ import annotations

import sys
import os
import time
import math
import argparse
from pathlib import Path
from typing import Optional

import cv2
import mss
import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt

# ─── Proje kök dizinini Python yoluna ekle ───────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]          # .../FIREFLY
sys.path.insert(0, str(_ROOT))

_TCTRACK_DIR = _ROOT / "src" / "python" / "vision" / "TCTrack"
sys.path.insert(0, str(_TCTRACK_DIR))
sys.path.insert(0, str(_ROOT / "src" / "python"))

# RF-DETR modülünden ihtiyaç duyulan parçalar
from src.python.vision.rfdetr_inference.run_rfdetr_tensorrt import (
    TRTInferencer,
    ScreenCapture,
    decode_outputs,
    apply_nms,
    CONF_THRESHOLD,
    NMS_IOU_THRESHOLD,
    CLASS_NAMES,
)

# TCTrack
from pysot.core.config import cfg
from pysot.models.utile_tctrack.model_builder import ModelBuilder_tctrack
from pysot.tracker.tctrack_tracker import TCTrackTracker
from pysot.utils.model_load import load_pretrain

# Route Intercept
from navigation.route_intercept_system import RoutePlanner, solve_intercept
from navigation.telemetry_conditioner import MeasurementGate, AltitudeLock
from navigation.orbit_lock import OrbitLockTracker
from guidance.route_follow_guidance import RouteFollowGuidance, GuidePointSlewLimiter
from guidance.pid_pitch import PitchController
from guidance.pid_throttle import ThrottleController
from guidance.pid_yaw import YawController
from route_viz import LiveDashboard, EllipseSimTarget

# ═══════════════════════════════════════════════════════════════════════════════
#  YAPILANDIRMA (GÖRSEL TAKİP)
# ═══════════════════════════════════════════════════════════════════════════════
ENGINE_PATH    = str(_ROOT / "object_detection" / "inference_model.engine")
TCTRACK_CONFIG = str(_TCTRACK_DIR / "experiments" / "TCTrack" / "config.yaml")
TCTRACK_SNAP   = str(_TCTRACK_DIR / "snapshot" / "general_model.pth")

MONITOR_INDEX  = 1
RESIZE_SCALE   = 0.5
OUTPUT_FPS     = 30.0
OUTPUT_PATH    = (
    f"/home/tom/Documents/Projeler/DogFight/"
    f"combined_rfdetr_tctrack_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
)

# Algılama Modeli Eşik Değeri
DET_CONF_THRESHOLD = 0.45

# Maksimum hedef alan oranı (kare alanına göre oran)
# Bu orandan daha büyük alan kaplayan hedefler takibe alınmaz.
MAX_TARGET_AREA_RATIO = 0.09

# Kilitlenme Dörtgeni Sınır Oranları ( Acquisition / Lock Zone )
LOCK_ROI_Y_BORDER = 0.10  # Üstten ve alttan bırakılacak boşluk oranı (%10)
LOCK_ROI_X_BORDER = 0.25  # Sağdan ve soldan bırakılacak boşluk oranı (%25)

# Döngü süresi (saniye)
DETECT_DURATION = 1.0   # RF-DETR ile tespit fazi suresi
TRACK_DURATION  = 4.0   # TCTrack takip fazi suresi

# TCTrack erken cikis esigi
TRACKER_LOST_SCORE = 0.70

# Görsel servo sırasında korunacak pitch açısı (derece, öne doğru)
SERVO_ATTACK_PITCH_DEG = 5.0

# ─── Renkler (BGR) ───────────────────────────────────────────────────────────
COL_DETR      = (0, 220, 90)    # yeşil  – tespit kutusu
COL_TRACK     = (0, 140, 255)   # turuncu – takip kutusu
COL_LOST      = (0, 0, 220)     # kırmızı – kayıp
COL_TEXT      = (255, 255, 255)
COL_BG        = (20, 20, 20)
COL_TIMER     = (0, 255, 220)   # timer çubuğu rengi
COL_LOCK_ZONE = (0, 255, 255)   # sarı – kilitlenme alanı

# ─── ROTA KİLİTLEME / HEDEF YAKALAMA YAPILANDIRMASI ──────────────────────────
DT_LOOP = 0.02
DRONE_SPEED_CMS = 2500.0  # Dron hızı: 25 m/s

# Simülasyon Ayarları (yalnızca --sim modunda geçerli)
SIM_TARGET_SPEED = 1800.0  # cm/s (18 m/s)
SIM_CENTER_X = 20000.0     # cm (200 m)
SIM_CENTER_Y = 20000.0     # cm (200 m)
SIM_A = 15000.0            # cm (150 m)
SIM_B = 10000.0            # cm (100 m)
SIM_GPS_NOISE = 500.0      # cm
SIM_GPS_PERIOD = 1.0       # s — hedef GPS örnekleme periyodu (yarışma: max 1 Hz)


# ═══════════════════════════════════════════════════════════════════════════════
#  KALMAN FİLTRESİ VE SERVOLAMA PID SINIFLARI
# ═══════════════════════════════════════════════════════════════════════════════
class SizePID:
    def __init__(self, Kp=0.003, Ki=0.0001, Kd=0.001):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_err = 0.0
        
    def calculate(self, err, dt):
        self.integral = max(-1.0, min(1.0, self.integral + err * dt))
        deriv = (err - self.prev_err) / dt if dt > 0 else 0.0
        self.prev_err = err
        output = self.Kp * err + self.Ki * self.integral + self.Kd * deriv
        return max(-0.8, min(0.8, output))
        
    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0


class ThrottlePID:
    def __init__(self, Kp=0.45, Ki=0.01, Kd=0.10):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_err = 0.0
        
    def calculate(self, err, dt):
        self.integral = max(-0.8, min(0.8, self.integral + err * dt))
        deriv = (err - self.prev_err) / dt if dt > 0 else 0.0
        self.prev_err = err
        output = self.Kp * err + self.Ki * self.integral + self.Kd * deriv
        return max(-0.9, min(0.9, output))
        
    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0


class YawPID:
    def __init__(self, Kp=0.40, Ki=0.01, Kd=0.08):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_err = 0.0
        
    def calculate(self, err, dt):
        self.integral = max(-0.5, min(0.5, self.integral + err * dt))
        deriv = (err - self.prev_err) / dt if dt > 0 else 0.0
        self.prev_err = err
        output = self.Kp * err + self.Ki * self.integral + self.Kd * deriv
        return max(-0.6, min(0.6, output))
        
    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0


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


def run_detr(
    inferencer: TRTInferencer,
    frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RF-DETR ile tam frame inference + NMS. (xyxy, confs, class_ids) döner."""
    h, w = frame.shape[:2]
    outputs = inferencer.infer(frame)
    xyxy, confs, class_ids = decode_outputs(outputs, h, w, DET_CONF_THRESHOLD)
    if len(xyxy) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )
    keep = apply_nms(xyxy, confs, class_ids, DET_CONF_THRESHOLD * 0.5, NMS_IOU_THRESHOLD)
    if len(keep) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )
    
    # Alanı çok büyük olan hedefleri filtrele
    valid_xyxy = xyxy[keep]
    valid_confs = confs[keep]
    valid_class_ids = class_ids[keep]

    filtered_indices = []
    total_area = h * w
    for idx, box in enumerate(valid_xyxy):
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        area = bw * bh
        if area / total_area <= MAX_TARGET_AREA_RATIO:
            filtered_indices.append(idx)

    if len(filtered_indices) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )
    return valid_xyxy[filtered_indices], valid_confs[filtered_indices], valid_class_ids[filtered_indices]


def connect_sdk():
    import drone_sdk as drone
    print("  drone_sdk'ya bağlanılıyor (127.0.0.1:12345) ...")
    while True:
        if drone.connect():
            print("  ✓ drone_sdk Bağlantısı kuruldu!\n")
            return drone
        print("  ✗ Bağlantı yok — 2 saniye sonra tekrar denenecek ...")
        time.sleep(2.0)


def write_report(log_rows, log_dir):
    """Headless koşu verisini CSV'ye yazar, özet metrikleri basar ve özet grafiği PNG olarak kaydeder."""
    if not log_rows:
        print("  (rapor) Kayıtlı veri yok.")
        return

    cols = ["t", "dx", "dy", "dz", "tx", "ty", "tz", "locked_alt",
            "dist_2d", "dist_3d", "yaw_err", "heading", "yaw_rate",
            "d_speed", "v_des", "cmd_yaw", "cmd_pitch", "cmd_thr", "cmd_roll",
            "meas_changed", "data_stale", "near_field", "orbit_locked",
            "true_dist", "d_pitch"]
    arr = np.array(log_rows, dtype=float)
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "combined_route_intercept_run.csv")
    np.savetxt(csv_path, arr, delimiter=",", header=",".join(cols), comments="")

    t = arr[:, 0]
    c = {name: arr[:, i] for i, name in enumerate(cols)}

    # Özet metrikler
    dur = t[-1] - t[0]
    dh = np.diff(c["heading"])
    dh = (dh + 180.0) % 360.0 - 180.0
    revs = float(np.sum(np.abs(dh)) / 360.0)
    revs_net = float(abs(np.sum(dh)) / 360.0)
    yaw_rate_rms = float(np.sqrt(np.mean(c["yaw_rate"] ** 2)))
    spd = c["d_speed"]
    alt_err = c["locked_alt"] - c["dz"]
    lock_mask = c["orbit_locked"] > 0.5
    seg = lock_mask if lock_mask.any() else np.ones_like(t, dtype=bool)

    print("\n  " + "=" * 56)
    print("  KOŞU RAPORU")
    print(f"    Süre                : {dur:8.1f} s   ({len(t)} örnek)")
    print(f"    Toplam gövde dönüşü : {revs:8.2f} tur (birikimli |Δyaw|/360)")
    print(f"    Net dönüş (rota)    : {revs_net:8.2f} tur — fazlalık {revs - revs_net:.2f} tur savrulma")
    print(f"    Yaw hızı RMS        : {yaw_rate_rms:8.1f} deg/s")
    print(f"    Drone hızı          : ort {np.mean(spd):6.0f} / std {np.std(spd):5.0f} cm/s")
    print(f"    İrtifa hatası RMS   : {np.sqrt(np.mean(alt_err**2)):8.0f} cm")
    print(f"    Guide mesafesi (kilit sonrası): ort {np.mean(c['dist_2d'][seg]):6.0f}"
          f" / min {np.min(c['dist_2d'][seg]):6.0f} cm")
    tgt_dist = np.hypot(c["tx"] - c["dx"], c["ty"] - c["dy"])
    print(f"    HEDEF mesafesi (kilit sonrası): ort {np.mean(tgt_dist[seg]):6.0f}"
          f" / min {np.min(tgt_dist[seg]):6.0f} cm")
    truth_ok = c["true_dist"] >= 0.0
    if (truth_ok & seg).any():
        m = truth_ok & seg
        print(f"    GERÇEK mesafe (debug, kilit sonrası): ort {np.mean(c['true_dist'][m]):6.0f}"
              f" / min {np.min(c['true_dist'][m]):6.0f} cm")
    print(f"    Bayat veri oranı    : {100.0*np.mean(c['data_stale']):8.1f} %")
    print(f"    Yakın alan oranı    : {100.0*np.mean(c['near_field']):8.1f} %")
    print(f"    CSV                 : {csv_path}")

    # Özet grafik
    plt.switch_backend("Agg")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Combined Target Intercept — Headless Koşu Raporu", fontweight="bold")

    ax = axes[0, 0]
    ax.plot(c["tx"], c["ty"], ".", ms=2, color="tab:red", alpha=0.4, label="Hedef GPS")
    ax.plot(c["dx"], c["dy"], color="tab:blue", lw=1.2, label="Drone")
    ax.set_title("XY Yörünge"); ax.set_xlabel("X (cm)"); ax.set_ylabel("Y (cm)")
    ax.axis("equal"); ax.legend(); ax.grid(True, alpha=0.4)

    ax = axes[0, 1]
    ax.plot(t, c["dz"], color="tab:blue", label="Drone Z")
    ax.plot(t, c["tz"], ".", ms=2, color="tab:red", alpha=0.4, label="Hedef Z (ham)")
    ax.plot(t, c["locked_alt"], color="tab:green", label="Kilitli ort. irtifa")
    ax.set_title("İrtifa"); ax.set_xlabel("t (s)"); ax.set_ylabel("Z (cm)")
    ax.legend(); ax.grid(True, alpha=0.4)

    ax = axes[1, 0]
    ax.plot(t, c["dist_2d"], color="tab:orange", label="2B mesafe (guide)")
    ax2 = ax.twinx()
    ax2.plot(t, c["d_speed"], color="tab:cyan", lw=0.8, label="Drone hızı")
    ax2.plot(t, c["v_des"], color="tab:green", lw=0.8, ls="--", label="İstenen hız")
    ax.set_title("Mesafe ve Hız"); ax.set_xlabel("t (s)")
    ax.set_ylabel("Mesafe (cm)"); ax2.set_ylabel("Hız (cm/s)")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.4)

    ax = axes[1, 1]
    ax.plot(t, c["yaw_err"], color="tab:red", lw=0.8, label="Yaw hatası (°)")
    ax2 = ax.twinx()
    ax2.plot(t, c["cmd_yaw"], color="tab:purple", lw=0.8, label="Yaw cmd")
    ax2.plot(t, c["cmd_pitch"], color="tab:pink", lw=0.8, label="Pitch cmd")
    ax2.plot(t, c["cmd_thr"], color="tab:cyan", lw=0.8, label="Thr cmd")
    ax2.set_ylim(-1.1, 1.1)
    ax.set_title("Yaw Hatası ve Komutlar"); ax.set_xlabel("t (s)")
    ax.set_ylabel("Yaw hatası (°)"); ax2.set_ylabel("Komut [-1..1]")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    png_path = os.path.join(log_dir, "combined_route_intercept_run.png")
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    print(f"    Rapor PNG           : {png_path}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  ANA KOŞU DÖNGÜSÜ
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="RF-DETR + TCTrack & Route Intercept Combined System")
    parser.add_argument("--sim", action="store_true", help="Yakalama simülasyonu çalıştırır.")
    parser.add_argument("--headless-intercept", action="store_true",
                        help="Rota ekranı GUI açmadan koşar; CSV + rapor PNG üretir.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Koşu süresi (s); dolunca otomatik durur.")
    parser.add_argument("--log-dir", default=None,
                        help="Headless rapor/CSV çıktı dizini (varsayılan: çalışma dizini).")
    args = parser.parse_args()
    # Video kayıt dizininin varlığından emin ol
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    # ── GÖRSEL TAKİP MODELLERİNİ YÜKLE ───────────────────────────────────────
    print("[INFO] RF-DETR TRT engine yukleniyor...")
    inferencer = TRTInferencer(ENGINE_PATH)
    inferencer.warmup(n=3)

    print("[INFO] TCTrack modeli yukleniyor...")
    tc_tracker, hp = load_tctrack()
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

    # ── OpenCV Hedef Ekranı Penceresi ─────────────────────────────────────────
    win_name = "Target Acquisition Window"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Video yazıcı dizinini oluştur
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    video_writer: Optional[cv2.VideoWriter] = None

    # Görsel takip durum değişkenleri
    prev_time = time.perf_counter()
    fps = 0.0
    fps_alpha = 0.9

    phase_start = time.perf_counter()  # mevcut fazın başlama anı
    in_track = False                   # False → TESPIT, True → TAKIP
    track_xyxy: Optional[np.ndarray] = None
    track_lost = False

    # ═════════════════════════════════════════════════════════════════════════
    #  SİMÜLASYON MODU (--sim)
    # ═════════════════════════════════════════════════════════════════════════
    if args.sim:
        print("\n" + "=" * 60)
        print("  Route Intercept — Combined Yakalama Simülasyonu")
        print(f"  Hedef Hızı: {SIM_TARGET_SPEED/100:.1f} m/s | Dron Hızı: {DRONE_SPEED_CMS/100:.1f} m/s")
        print("=" * 60)

        # Rota Dashboard Penceresi
        dash = LiveDashboard(
            "Combined Intercept System — Sim Map",
            "Otonom Yakalama Haritası",
            target_label="Hedef GPS Noktaları")

        planner = RoutePlanner(buffer_size=60, min_fit_points=15, residual_threshold=1500.0)
        ellipse_sim = EllipseSimTarget(SIM_CENTER_X, SIM_CENTER_Y, SIM_A, SIM_B,
                                       SIM_TARGET_SPEED, SIM_GPS_NOISE)

        target_x_log, target_y_log = [], []
        drone_x_log, drone_y_log = [0.0], [0.0]
        t_hist, dist_log, res_log = [], [], []

        t = 0.0
        intercept_resolved = False
        step = 0
        last_gps_t = -1e9

        try:
            while t < (args.duration if args.duration is not None else 80.0):
                step_start = time.perf_counter()

                # ── 1. GÖRSEL TAKİP (OpenCV) ──────────────────────────────────
                frame = capture.get()
                if frame is None:
                    continue
                frame = frame.copy()
                h, w = frame.shape[:2]

                # FPS hesabı
                now = time.perf_counter()
                dt_fps = now - prev_time
                if dt_fps > 0:
                    cur_fps = 1.0 / dt_fps
                    fps = fps_alpha * fps + (1.0 - fps_alpha) * cur_fps if fps > 0 else cur_fps
                prev_time = now

                elapsed = now - phase_start
                xyxy = []

                if not in_track:
                    phase = "TESPIT"
                    xyxy, confs, class_ids = run_detr(inferencer, frame)
                    for box, conf, cid in zip(xyxy, confs, class_ids):
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), COL_DETR, 2)
                        lbl = f"{CLASS_NAMES.get(int(cid), str(cid))} {conf:.2f}"
                        put_text_bg(frame, lbl, (x1, y2 + 14), scale=0.45, fg=COL_DETR, bg=COL_BG)

                    # Tüm algılamaları kilit alanı kısıtı olmadan kabul et (dron hedefi merkeze çekecektir)
                    valid_detections = []
                    for box, conf, cid in zip(xyxy, confs, class_ids):
                        valid_detections.append((box, conf, cid))

                    if len(valid_detections) > 0:
                        best_i = int(np.argmax([d[1] for d in valid_detections]))
                        init_box, best_conf, init_cls = valid_detections[best_i]
                        init_xywh = xywh_from_xyxy(init_box)

                        tc_tracker.init(frame, init_xywh)
                        in_track = True
                        track_lost = False
                        track_xyxy = init_box.copy()

                        cls_name = CLASS_NAMES.get(int(init_cls), str(init_cls))
                        print(f"[INFO] TAKIP fazina gecildi — {cls_name} conf={best_conf:.2f} bbox={init_xywh}")
                        phase_start = time.perf_counter()
                    else:
                        draw_timer_bar(frame, elapsed, DETECT_DURATION, "TESPIT", COL_DETR)
                        if elapsed >= DETECT_DURATION:
                            print("[WARN] Yeterli tespit toplanamadi (0 kare). Tespit fazi yenilendi.")
                            phase_start = time.perf_counter()

                else:
                    phase = "TAKIP"
                    tc_out = tc_tracker.track(frame, hp)
                    tc_score = float(tc_out["best_score"])
                    tc_xyxy = xyxy_from_tc(tc_out["bbox"])

                    track_lost = tc_score < TRACKER_LOST_SCORE

                    if not track_lost:
                        bw = tc_xyxy[2] - tc_xyxy[0]
                        bh = tc_xyxy[3] - tc_xyxy[1]
                        if (bw * bh) / (h * w) > MAX_TARGET_AREA_RATIO:
                            track_lost = True

                    if not track_lost:
                        track_xyxy = tc_xyxy.copy()
                        x1, y1, x2, y2 = int(tc_xyxy[0]), int(tc_xyxy[1]), int(tc_xyxy[2]), int(tc_xyxy[3])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), COL_TRACK, 3)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        cv2.drawMarker(frame, (cx, cy), COL_TRACK, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
                        put_text_bg(frame, f"TC: {tc_score:.2f}", (x1, y1 - 2), scale=0.6, fg=COL_TRACK)
                    else:
                        if track_xyxy is not None:
                            overlay = frame.copy()
                            x1, y1, x2, y2 = (int(v) for v in track_xyxy)
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), COL_LOST, 2)
                            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
                        put_text_bg(frame, "HEDEF KAYBOLDU", (w // 2 - 90, h // 2), scale=0.8, fg=COL_LOST)

                    # Süresiz takip durumunda durum çubuğu çizilir
                    cv2.rectangle(frame, (0, 0), (w, 8), COL_TRACK, -1)
                    put_text_bg(frame, "TAKIP AKTIF — HEDEFE KILITLENILDI", (10, 26), scale=0.6, fg=COL_TRACK)

                    if track_lost:
                        print(f"[INFO] TESPIT fazina donuluyor (hedef kayboldu).")
                        in_track = False
                        track_xyxy = None
                        track_lost = False
                        phase_start = time.perf_counter()

                n_det = 1 if in_track else len(xyxy)
                draw_hud(frame, fps, phase, n_det)

                # Kilitlenme Dörtgeni Çizimi
                lx1 = int(LOCK_ROI_X_BORDER * w)
                ly1 = int(LOCK_ROI_Y_BORDER * h)
                lx2 = int((1.0 - LOCK_ROI_X_BORDER) * w)
                ly2 = int((1.0 - LOCK_ROI_Y_BORDER) * h)
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), COL_LOCK_ZONE, 2)
                put_text_bg(frame, "KILIT ALANI", (lx1 + 8, ly1 + 22), scale=0.55, fg=COL_LOCK_ZONE)

                # Video kayıt
                if video_writer is None:
                    oh, ow = frame.shape[:2]
                    video_writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), OUTPUT_FPS, (ow, oh))
                if video_writer:
                    # Gerçek geçen süreye göre kareyi çoğaltarak yaz (kare atlamasını/hızlanmasını önler)
                    n_write = max(1, int(round(DT_LOOP * OUTPUT_FPS)))
                    for _ in range(n_write):
                        video_writer.write(frame)

                cv2.imshow(win_name, frame)

                # ── 2. ROTA INTERCEPT SİMÜLASYONU ──────────────────────────────
                tx_real, ty_real = ellipse_sim.truth(t)

                if t - last_gps_t >= SIM_GPS_PERIOD:
                    tx_noisy, ty_noisy = ellipse_sim.measure(t)
                    last_gps_t = t
                    target_x_log.append(tx_noisy)
                    target_y_log.append(ty_noisy)
                    planner.add_observation(t, tx_noisy, ty_noisy)

                dx, dy = drone_x_log[-1], drone_y_log[-1]

                intercept_pt = None
                if planner.route.fitted:
                    sol = solve_intercept(planner.route, t, (dx, dy), DRONE_SPEED_CMS)
                    if sol is not None:
                        intercept_pt = sol['intercept_point']
                        ix, iy = intercept_pt
                        v_dx, v_dy = ix - dx, iy - dy
                        v_dist = math.hypot(v_dx, v_dy)
                        if v_dist > 100.0:
                            step_dist = min(DRONE_SPEED_CMS * DT_LOOP, v_dist)
                            dx += (v_dx / v_dist) * step_dist
                            dy += (v_dy / v_dist) * step_dist
                        if v_dist < 200.0 and not intercept_resolved:
                            print(f"\n  [YAKALANDI!] Hedef t={t:.2f} saniyede elips üzerinde yakalandı.")
                            intercept_resolved = True
                    dash.set_ellipse(planner.route)

                drone_x_log.append(dx)
                drone_y_log.append(dy)

                t_hist.append(t)
                dist_log.append(math.hypot(tx_real - dx, ty_real - dy))
                res_log.append(planner.last_residual if planner.last_residual is not None else 0.0)

                step += 1
                if step % 10 == 0:
                    dash.update((target_x_log, target_y_log), (drone_x_log, drone_y_log),
                                tx_real, ty_real, dx, dy, t_hist, dist_log, res_log, intercept_pt)
                    plt.pause(0.001)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    print("[INFO] Manuel sifirlama — TESPIT fazina donuluyor.")
                    in_track = False
                    track_xyxy = None
                    track_lost = False
                    phase_start = time.perf_counter()

                t += DT_LOOP

                elapsed_time = time.perf_counter() - step_start
                sleep_t = DT_LOOP - elapsed_time
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            print("\nSimülasyon kesildi.")
        finally:
            capture.stop()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            dash.block()

    # ═════════════════════════════════════════════════════════════════════════
    #  CANLI KOŞU MODU (drone_sdk)
    # ═════════════════════════════════════════════════════════════════════════
    else:
        drone = connect_sdk()
        if drone is None:
            return

        print("  Canlı Rota ve Yakalama Çözücü Başlatıldı. CTRL+C ile durdurun.\n")

        # Rota Dashboard Penceresi
        dash = None
        if not args.headless_intercept:
            dash = LiveDashboard(
                "Combined Intercept System — Canlı Yakalama (drone_sdk)",
                "Canlı Yakalama Haritası")

        # Config'i yükle
        cfg_path = os.path.join(_ROOT, "config", "pid_params.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            pid_config = yaml.safe_load(f)

        yaw_ctrl = YawController(pid_config)
        pitch_ctrl = PitchController(pid_config)
        throttle_ctrl = ThrottleController(pid_config)
        guidance = RouteFollowGuidance(pid_config)

        # Görsel servo PID kontrolcüleri (Yumuşatılmış / Soften edilmiş kazançlar)
        size_pid = SizePID(Kp=0.003, Ki=0.0001, Kd=0.001)
        thr_pid = ThrottlePID(Kp=0.75, Ki=0.01, Kd=0.10)
        yaw_pid = YawPID(Kp=0.40, Ki=0.01, Kd=0.08)

        ri_cfg = pid_config.get("route_intercept", {})
        NOMINAL_TARGET_SPEED = float(ri_cfg.get("nominal_target_speed_cms", 1700.0))
        LOCK_STABLE_LAPS = int(ri_cfg.get("lock_stable_laps", 3))
        WAIT_FOR_LOCK = bool(ri_cfg.get("wait_for_lock", True))
        NEAR_FIELD_CM = 700.0

        gate = MeasurementGate(jump_reject_cm=50000.0, stale_timeout_s=2.5)
        alt_lock = AltitudeLock(buffer_size=30, jump_reject_cm=1500.0, reject_persist=8)
        orbit_tracker = OrbitLockTracker(
            buffer_size=60, min_fit_points=15, residual_threshold=1500.0,
            outlier_threshold_cm=5000.0, lock_stable_laps=LOCK_STABLE_LAPS,
            min_lap_points=15, period_coverage=0.0, verbose=True)
        guide_slew = GuidePointSlewLimiter(NOMINAL_TARGET_SPEED * 2.0)

        target_x_log, target_y_log = [], []
        drone_x_log, drone_y_log = [], []
        t_hist, dist_log, res_log = [], [], []

        wait_msg_shown = False
        last_yaw_cmd = 0.0
        prev_heading = None
        log_rows = []

        t0 = time.time()
        prev_loop_t = t0
        step = 0
        running = True

        try:
            while running:
                step_start = time.time()
                t_now = step_start - t0
                dt = max(0.005, min(0.1, step_start - prev_loop_t))
                prev_loop_t = step_start

                # ── 1. GÖRSEL TAKİP (OpenCV) ──────────────────────────────────
                frame = capture.get()
                if frame is None:
                    continue
                frame = frame.copy()
                h, w = frame.shape[:2]

                # FPS hesabı
                now = time.perf_counter()
                dt_fps = now - prev_time
                if dt_fps > 0:
                    cur_fps = 1.0 / dt_fps
                    fps = fps_alpha * fps + (1.0 - fps_alpha) * cur_fps if fps > 0 else cur_fps
                prev_time = now

                elapsed = now - phase_start
                xyxy = []

                if not in_track:
                    phase = "TESPIT"
                    xyxy, confs, class_ids = run_detr(inferencer, frame)
                    # Sadece 3. yörüngeden sonra görsel takip çalışsın
                    if orbit_tracker.orbit_count >= 3:
                        for box, conf, cid in zip(xyxy, confs, class_ids):
                            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), COL_DETR, 2)
                            lbl = f"{CLASS_NAMES.get(int(cid), str(cid))} {conf:.2f}"
                            put_text_bg(frame, lbl, (x1, y2 + 14), scale=0.45, fg=COL_DETR, bg=COL_BG)

                        # Tüm algılamaları kilit alanı kısıtı olmadan kabul et (dron hedefi merkeze çekecektir)
                        valid_detections = []
                        for box, conf, cid in zip(xyxy, confs, class_ids):
                            valid_detections.append((box, conf, cid))

                        if len(valid_detections) > 0:
                            best_i = int(np.argmax([d[1] for d in valid_detections]))
                            init_box, best_conf, init_cls = valid_detections[best_i]
                            init_xywh = xywh_from_xyxy(init_box)

                            tc_tracker.init(frame, init_xywh)
                            in_track = True
                            track_lost = False
                            track_xyxy = init_box.copy()

                            cls_name = CLASS_NAMES.get(int(init_cls), str(init_cls))
                            print(f"[INFO] TAKIP fazina gecildi — {cls_name} conf={best_conf:.2f} bbox={init_xywh}")
                            phase_start = time.perf_counter()
                        else:
                            draw_timer_bar(frame, elapsed, DETECT_DURATION, "TESPIT", COL_DETR)
                            if elapsed >= DETECT_DURATION:
                                print("[WARN] Yeterli tespit toplanamadi (0 kare). Tespit fazi yenilendi.")
                                phase_start = time.perf_counter()
                    else:
                        # Görsel takip pasif, yörünge kilitlenme durumu bekleniyor
                        put_text_bg(frame, f"GORSEL TAKIP PASIF (TUR: {orbit_tracker.orbit_count}/3)", 
                                    (10, 140), scale=0.6, fg=(0, 0, 255))
                        draw_timer_bar(frame, elapsed, DETECT_DURATION, "BEKLEMEDE", COL_DETR)
                        if elapsed >= DETECT_DURATION:
                            phase_start = time.perf_counter()

                else:
                    phase = "TAKIP"
                    tc_out = tc_tracker.track(frame, hp)
                    tc_score = float(tc_out["best_score"])
                    tc_xyxy = xyxy_from_tc(tc_out["bbox"])

                    track_lost = tc_score < TRACKER_LOST_SCORE

                    if not track_lost:
                        bw = tc_xyxy[2] - tc_xyxy[0]
                        bh = tc_xyxy[3] - tc_xyxy[1]
                        if (bw * bh) / (h * w) > MAX_TARGET_AREA_RATIO:
                            track_lost = True

                    if not track_lost:
                        track_xyxy = tc_xyxy.copy()
                        x1, y1, x2, y2 = int(tc_xyxy[0]), int(tc_xyxy[1]), int(tc_xyxy[2]), int(tc_xyxy[3])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), COL_TRACK, 3)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        cv2.drawMarker(frame, (cx, cy), COL_TRACK, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
                        put_text_bg(frame, f"TC: {tc_score:.2f}", (x1, y1 - 2), scale=0.6, fg=COL_TRACK)
                    else:
                        if track_xyxy is not None:
                            overlay = frame.copy()
                            x1, y1, x2, y2 = (int(v) for v in track_xyxy)
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), COL_LOST, 2)
                            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
                        put_text_bg(frame, "HEDEF KAYBOLDU", (w // 2 - 90, h // 2), scale=0.8, fg=COL_LOST)

                    # Süresiz takip durumunda durum çubuğu çizilir
                    cv2.rectangle(frame, (0, 0), (w, 8), COL_TRACK, -1)
                    put_text_bg(frame, "TAKIP AKTIF — HEDEFE KILITLENILDI", (10, 26), scale=0.6, fg=COL_TRACK)

                    if track_lost:
                        print(f"[INFO] TESPIT fazina donuluyor (hedef kayboldu).")
                        in_track = False
                        track_xyxy = None
                        track_lost = False
                        phase_start = time.perf_counter()

                n_det = 1 if in_track else len(xyxy)
                draw_hud(frame, fps, phase, n_det)

                # Kilitlenme Dörtgeni Çizimi
                lx1 = int(LOCK_ROI_X_BORDER * w)
                ly1 = int(LOCK_ROI_Y_BORDER * h)
                lx2 = int((1.0 - LOCK_ROI_X_BORDER) * w)
                ly2 = int((1.0 - LOCK_ROI_Y_BORDER) * h)
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), COL_LOCK_ZONE, 2)
                put_text_bg(frame, "KILIT ALANI", (lx1 + 8, ly1 + 22), scale=0.55, fg=COL_LOCK_ZONE)

                # Video kayıt
                if video_writer is None:
                    oh, ow = frame.shape[:2]
                    video_writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), OUTPUT_FPS, (ow, oh))
                if video_writer:
                    # Gerçek geçen süreye göre kareyi çoğaltarak yaz (kare atlamasını/hızlanmasını önler)
                    n_write = max(1, int(round(dt * OUTPUT_FPS)))
                    for _ in range(n_write):
                        video_writer.write(frame)

                cv2.imshow(win_name, frame)

                # ── 2. ROTA INTERCEPT CANLI SDK DÖNGÜSÜ ────────────────────────
                t_data = drone.get_telemetry()
                d_pos = t_data["drone"]["position"]
                d_rot = t_data["drone"]["rotation"]
                d_vel = t_data["drone"]["velocity"]
                tar_pos = t_data["target"]["position"]

                dx, dy, dz = d_pos[0], d_pos[1], d_pos[2]
                tx_raw, ty_raw, tz = tar_pos[0], tar_pos[1], tar_pos[2]

                if dx == 0.0 and dy == 0.0 and tx_raw == 0.0 and ty_raw == 0.0:
                    time.sleep(DT_LOOP)
                    continue

                g = gate.update(t_now, tx_raw, ty_raw)
                tx, ty = g.x, g.y
                meas_changed, data_valid, data_stale = g.meas_changed, g.data_valid, g.data_stale

                locked_alt = alt_lock.update(tz, meas_changed, data_valid)
                if locked_alt is None:
                    time.sleep(DT_LOOP)
                    continue

                target_x_log.append(tx)
                target_y_log.append(ty)
                drone_x_log.append(dx)
                drone_y_log.append(dy)

                active_route = orbit_tracker.step(t_now, tx, ty, meas_changed)

                intercept_pt = None
                guide_x, guide_y = tx, ty
                ring_v_des = None
                if active_route.fitted:
                    guide_x, guide_y, ring_v_des = guidance.guide(active_route, dx, dy, tx, ty, dt=dt)
                    intercept_pt = (float(guide_x), float(guide_y))
                    if dash is not None:
                        dash.set_ellipse(active_route)

                t_hist.append(t_now)
                dist_log.append(math.hypot(tx - dx, ty - dy))
                res_log.append(orbit_tracker.last_residual if orbit_tracker.last_residual is not None else 0.0)

                guide_x, guide_y = guide_slew.update(guide_x, guide_y, dt)

                ldx = guide_x - dx
                ldy = guide_y - dy
                ldz = locked_alt - dz
                dist_2d = math.hypot(ldx, ldy)
                dist_3d = math.hypot(dist_2d, ldz)

                heading = d_rot[2]
                if prev_heading is None:
                    yaw_rate_meas = 0.0
                else:
                    dh = heading - prev_heading
                    while dh > 180:
                        dh -= 360
                    while dh < -180:
                        dh += 360
                    yaw_rate_meas = dh / dt
                prev_heading = heading

                near_field = dist_2d < NEAR_FIELD_CM
                # Hedef takibi (interception) 3. yörüngeden sonra (tur #3) çalışmaya başlasın
                waiting = WAIT_FOR_LOCK and (orbit_tracker.orbit_count < 3)
                if waiting and not wait_msg_shown:
                    print("  ⏸ Rota öğreniliyor (3 yörünge kilidi bekleniyor) — hover'da bekleniyor...")
                    wait_msg_shown = True
                elif not waiting and wait_msg_shown:
                    print("  ▶ Yörünge kilitli (3 tur tamamlandı) — takibe geçiliyor.")
                    wait_msg_shown = False

                yaw_error, yaw_cmd = yaw_ctrl.calculate(ldx, ldy, d_rot[2], dt, yaw_rate=yaw_rate_meas)

                if waiting or data_stale or near_field:
                    yaw_cmd = 0.0
                    yaw_ctrl.reset()
                    last_yaw_cmd = 0.0
                    pitch_align_err = 0.0
                elif not data_valid:
                    yaw_cmd = last_yaw_cmd
                    pitch_align_err = yaw_error
                else:
                    last_yaw_cmd = yaw_cmd
                    pitch_align_err = yaw_error

                v_des = guidance.desired_speed(ring_v_des, dist_2d, waiting, data_stale)

                hd_rad = math.radians(d_rot[2])
                v_fwd = d_vel[0] * math.cos(hd_rad) + d_vel[1] * math.sin(hd_rad)
                target_elevation = math.degrees(math.atan2(ldz, max(dist_2d, 1.0)))
                smooth_pitch, pitch_cmd = pitch_ctrl.calculate(
                    state="TAKIP",
                    target_speed_cmd=v_des,
                    drone_speed=v_fwd,
                    target_elevation=target_elevation,
                    drone_pitch=d_rot[1],
                    yaw_error=pitch_align_err,
                    dt=dt,
                )

                thr_cmd = throttle_ctrl.calculate(ldz, dt)

                fy = max(-1.0, min(1.0, yaw_cmd))
                fp = -max(-1.0, min(1.0, pitch_cmd))
                f_thr = max(-1.0, min(1.0, thr_cmd))

                # Eğer görsel takip aktifse ve kilit kaybolmadıysa karışım (blending) uygula
                is_servo_active = in_track and (not track_lost) and (track_xyxy is not None)
                
                if is_servo_active:
                    # Kamera ekran merkezi ve hedef merkezi hesabı
                    tc_cx = (track_xyxy[0] + track_xyxy[2]) / 2.0
                    tc_cy = (track_xyxy[1] + track_xyxy[3]) / 2.0
                    
                    scx = w / 2.0
                    scy = h / 2.0
                    
                    # Piksel hataları (normalleştirilmiş, -1 ile 1 arası)
                    err_x = (tc_cx - scx) / scx
                    err_y = (scy - tc_cy) / scy  # Pozitif = hedef yukarıda
                    
                    # Hedef boyutunun kontrolü (küçülürse hızlanma PIDs)
                    bw = track_xyxy[2] - track_xyxy[0]
                    bh = track_xyxy[3] - track_xyxy[1]
                    current_size = max(bw, bh)
                    
                    target_touch_size = 300.0
                    err_size = target_touch_size - current_size
                    
                    # Servolama komutları
                    fy_servo = yaw_pid.calculate(err_x, dt)
                    f_thr_servo = thr_pid.calculate(err_y, dt)
                    
                    # Görsel takip sırasında öne doğru sabit pitch açısını koru (örn. 5 derece)
                    # 60 derece tam ölçek olduğundan, stick değeri = SERVO_ATTACK_PITCH_DEG / 60.0
                    fp_servo_speed = SERVO_ATTACK_PITCH_DEG / 60.0
                    fp_servo = -fp_servo_speed  # Simulator için öne doğru pitch negatif stick değeridir
                    
                    # Rotadan sapma kontrolü (lerp)
                    deviation = 0.0
                    if active_route.fitted:
                        nx, ny = active_route.point_on_ellipse(active_route.nearest_phi(dx, dy))
                        deviation = math.hypot(dx - nx, dy - ny)
                    
                    # Sapma < 500 cm ise tamamen görsel servo (%100)
                    # Sapma > 1500 cm ise tamamen rotaya dönüş (%0 görsel servo)
                    if deviation <= 500.0:
                        blend_factor = 1.0
                    elif deviation >= 1500.0:
                        blend_factor = 0.0
                    else:
                        blend_factor = (1500.0 - deviation) / 1000.0
                        
                    fy_final = blend_factor * fy_servo + (1.0 - blend_factor) * fy
                    fp_final = blend_factor * fp_servo + (1.0 - blend_factor) * fp
                    f_thr_final = blend_factor * f_thr_servo + (1.0 - blend_factor) * f_thr
                    
                    # Dokunma Kontrolü ve HUD üzerine bilgi yazdırma
                    if current_size >= target_touch_size:
                        cv2.putText(frame, "TOUCHED!", (int(w//2 - 100), int(h//2)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4, cv2.LINE_AA)
                        print("[TOUCH] Hedefe dokunuldu! Otonom gorev basariyla tamamlandi.")
                else:
                    # Görsel takip aktif değilse PIDs resetle
                    yaw_pid.reset()
                    thr_pid.reset()
                    size_pid.reset()
                    
                    fy_final = fy
                    fp_final = fp
                    f_thr_final = f_thr
                    blend_factor = 0.0

                # Komutları gönder
                drone.set_control_surfaces(f_thr_final, fp_final, 0.0, fy_final, True)

                if args.headless_intercept:
                    truth = drone.get_debug_truth()
                    if truth.get("available"):
                        tp = truth["target"]["position"]
                        true_dist = math.hypot(tp[0] - dx, tp[1] - dy)
                    else:
                        true_dist = -1.0
                    log_rows.append((
                        t_now, dx, dy, dz, tx, ty, tz, locked_alt,
                        dist_2d, dist_3d, yaw_error, d_rot[2], yaw_rate_meas,
                        v_fwd, v_des, fy, fp, f_thr, 0.0,
                        int(meas_changed), int(data_stale), int(near_field),
                        int(orbit_tracker.orbit_locked), true_dist,
                        d_rot[1],
                    ))

                if args.duration is not None and t_now >= args.duration:
                    print(f"\n  Süre doldu ({args.duration:.0f} s) — koşu bitiriliyor.")
                    break

                step += 1
                if dash is not None and step % 5 == 0 and target_x_log:
                    dash.update((target_x_log, target_y_log), (drone_x_log, drone_y_log),
                                tx, ty, dx, dy, t_hist, dist_log, res_log, intercept_pt)
                    plt.pause(0.001)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    print("[INFO] Manuel sifirlama — TESPIT fazina donuluyor.")
                    in_track = False
                    track_xyxy = None
                    track_lost = False
                    phase_start = time.perf_counter()

                elapsed_time = time.time() - step_start
                sleep_t = DT_LOOP - elapsed_time
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            print("\n  Canlı takip sonlandırıldı.")
        finally:
            try:
                drone.set_control_surfaces(0.0, 0.0, 0.0, 0.0, True)
            except Exception:
                pass
            drone.disconnect()
            capture.stop()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            if args.headless_intercept:
                write_report(log_rows, args.log_dir or os.getcwd())
            elif dash is not None:
                dash.block()


if __name__ == "__main__":
    main()
