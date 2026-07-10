"""
RF-DETR TensorRT + SAHI + SORT-CSRT + Merkez Kilitlenme + Gorsel Servolama Testi
=================================================================================

Pipeline:
  Ekran Yakalama -> SAHI Tiling -> RF-DETR (TensorRT) -> Class-aware NMS
      -> SORT-CSRT Tracker -> Merkez Kilitlenme Dortgeni -> [G tusu] Gorsel Servo

Faz Sistemi (test_rfdetr_sahi_csrt.py ile ayni prensip):
  TESPIT FAZI  (DETECT_DURATION sn)
    - SAHI dilimlemeli RF-DETR TensorRT her kareyi analiz eder.
    - Tespitler SORT-CSRT'ye beslenir.
  TAKIP FAZI   (TRACK_DURATION sn)
    - RF-DETR CALISMAZ -> yuksek FPS; SORT-CSRT (CSRT prediction) takip eder.
    - Tum tracker'lar kaybolursa erken TESPIT fazina doner.

TAKIP fazi hizlandirmalari (yapilandirma blogundan ayarlanir):
    - TRACK_ONLY_LOCK_TARGET: faza girerken kilit hedefi disindaki tracker'lar
      atilir (CSRT maliyeti tracker sayisiyla dogrusal buyur).
    - TRACK_STRIDE: CSRT her N karede bir kosulur, arada son kutu kullanilir.
    - CSRT_FAST: hafifletilmis CSRT (olcek 33->9, ADMM 4->2, sablon 200->150).
    - Yinelenen kare atlama: ekran yakalama ayni kareyi verdiyse islenmez.

Kilitlenme:
  - Ekran merkezinde sabit bir KILITLENME DORTGENI cizilir.
  - Merkeze en yakin takip edilen hedef secilir (secim track ID'ye yapiskan).
  - Hedef merkezi dortgen icinde LOCK_CONFIRM_FRAMES kare kalirsa -> KILIT.
  - Kilit, histerezisli (buyutulmus) dortgenden cikinca dusner.

Gorsel Servolama (G tusu ile ac/kapa):
  - Piksel hatasi kamera FOV'u (125 derece yatay) ile aciya cevrilir.
  - Hata LPF (tau) ile yumusatilir; PD + slew limit + devreye giris rampasi
    ile drone'a yumusak yaw / throttle / pitch komutlari gonderilir
    (drone_sdk.set_control_surfaces). Simulator bagli degilse servo acilamaz.
  - Servo KAPALI iken hicbir komut gonderilmez -> sadece goruntu isleme + kilit.
  - Hedef kaybolursa komutlar sifira dogru sondurulur; LOST_TIMEOUT sonrasi
    servo kendini kapatir.

Kullanim:
  cd /home/tom/Documents/FIREFLY
  python3 tests/python/test_rfdetr_trt_visual_servo.py

Kontroller:
  G : Gorsel servolamayi ac/kapa (yonelim algoritmasi)
  R : Tracker'lari sifirla (TESPIT fazina don)
  Q : Cikis

Gereksinimler:
  - tensorrt + pycuda (TRT engine: ENGINE_PATH)
  - opencv-contrib-python (CSRT), numpy, mss, scipy (veya lap)
  - Servo icin: DronesOfWar simulatoru TCP :12345'te calisiyor olmali
"""

from __future__ import annotations

import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ─── Proje yollarini ekle ─────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]            # .../FIREFLY
sys.path.insert(0, str(_ROOT))

# sort_csrt.py `from skimage import io` kullaniyor ama gerekli degil;
# skimage kurulu degilse sahte modul ile import hatasini atla.
if "skimage" not in sys.modules:
    try:
        import skimage  # noqa: F401
    except ImportError:
        _sk = types.ModuleType("skimage")
        _sk.io = types.ModuleType("skimage.io")         # type: ignore[attr-defined]
        sys.modules["skimage"] = _sk
        sys.modules["skimage.io"] = _sk.io              # type: ignore[attr-defined]

_SORT_DIR = _ROOT / "src" / "python" / "vision" / "sort_csrt_tracking"
_TRT_DIR  = _ROOT / "src" / "python" / "vision" / "rfdetr_inference"
sys.path.insert(0, str(_ROOT / "src" / "python"))
sys.path.insert(0, str(_SORT_DIR))
sys.path.insert(0, str(_TRT_DIR))

from sort_csrt import Sort, CSRTTracker                 # noqa: E402

# TRT inference altyapisi run_rfdetr_tensorrt.py'den yeniden kullanilir
# (TRTInferencer, sahi_predict, ScreenCapture, CLASS_NAMES).
import run_rfdetr_tensorrt as rt                        # noqa: E402

import drone_sdk                                        # noqa: E402

# Kilitlenme + gorsel servo cekirdegi (CUDA'ya bagimsiz, yeniden kullanilabilir)
from vision.visual_servo import (                       # noqa: E402
    ServoConfig, LockOnManager, VisualServoController,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  YAPILANDIRMA
# ═══════════════════════════════════════════════════════════════════════════════

# ── Model / Kaynak ────────────────────────────────────────────────────────────
ENGINE_PATH    = str ("object_detection/inference_model.engine")
CONF_THRESHOLD = 0.70
MONITOR_INDEX  = 1            # 1 = ana ekran
RESIZE_SCALE   = 0.4          # ekran boyutu carpani (inference yuku icin)

# ── SAHI dilimleme (rt modul sabitlerini ezer) ────────────────────────────────
SLICE_WIDTH       = 640
SLICE_HEIGHT      = 640
OVERLAP_RATIO     = 0.2
NMS_IOU_THRESHOLD = 0.45

# ── Faz sureleri (saniye) ─────────────────────────────────────────────────────
DETECT_DURATION = 1.5    # SAHI + RF-DETR tespit fazi
TRACK_DURATION  = 4.0    # sadece CSRT takip fazi

# ── SORT-CSRT ─────────────────────────────────────────────────────────────────
SORT_MAX_AGE    = 300
SORT_MIN_HITS   = 1
SORT_IOU_THRESH = -0.8   # GIoU esigi (negatif = esnek esleme)

# ── TAKIP fazi hizlandirma ────────────────────────────────────────────────────
# CSRT maliyeti = tracker sayisi x parametre agirligi x kare orani. Uc kol da
# ayri ayri kisilir; toplam kazanc carpimsaldir.
TRACK_ONLY_LOCK_TARGET = True   # TAKIP'e girerken kilit hedefi disindaki tracker'lari at
TRACK_STRIDE           = 2      # CSRT guncellemesi her N karede bir (1 = her kare)
CSRT_FAST              = True   # hafifletilmis CSRT parametreleri
CSRT_NUM_SCALES        = 9      # varsayilan 33 — olcek arama sayisi
CSRT_ADMM_ITERS        = 2      # varsayilan 4  — filtre optimizasyon iterasyonu
CSRT_TEMPLATE_SIZE     = 150.0  # varsayilan 200 — sablon cozunurlugu
RECORD_VIDEO           = True   # False = video yazimini kapat (birkac ms/kare)

# ── Kilitlenme dortgeni ───────────────────────────────────────────────────────
LOCK_BOX_FRAC       = 0.22   # dortgen kenari = min(w, h) * frac
LOCK_HYST_SCALE     = 1.6    # kilidi dusurme dortgeni (histerezis) carpani
LOCK_CONFIRM_FRAMES = 8      # kilit onayi icin ardisik kare
LOCK_RELEASE_FRAMES = 12     # kilidi dusurmek icin ardisik "disarida" kare

# ── Kamera geometrisi (SDK sabitleri: 125 derece FOV, 25 derece yukari tilt) ──
CAMERA_FOV_H_DEG = 125.0

# ── Gorsel servo kazanclari ───────────────────────────────────────────────────
SERVO_KP_YAW      = 0.022   # stick / derece
SERVO_KD_YAW      = 0.012   # stick / (derece/s) — filtreli hata turevi
SERVO_KP_VERT     = 0.020   # throttle / derece (dikey hiz komutu)
SERVO_PITCH_MAX   = 0.25    # hedef merkezdeyken ileri yaklasma pitch'i
SERVO_YAW_MAX     = 0.60
SERVO_THR_MAX     = 0.50
SERVO_ERR_TAU     = 0.12    # hata LPF zaman sabiti (s) — yumusatma
SERVO_RAMP_S      = 1.5     # devreye giris rampasi (s) — ani sicramayi onler
SERVO_SLEW        = {"yaw": 2.0, "pitch": 0.8, "throttle": 1.5}  # stick/s
SERVO_LOST_TIMEOUT = 1.5    # hedef kayipken servo'nun kendini kapatma suresi (s)
YAW_SIGN      = +1.0        # isaret ters gelirse -1 yap
THROTTLE_SIGN = +1.0

# ── Cikti ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_FPS = 30.0

# ── Renkler (BGR) ─────────────────────────────────────────────────────────────
COL_DETECT   = (0, 220, 90)      # yesil  — ham tespit
COL_SEARCH   = (0, 0, 220)       # kirmizi — kilit yok
COL_ACQUIRE  = (0, 200, 255)     # sari    — kilit onaylaniyor
COL_LOCKED   = (0, 255, 0)       # yesil   — KILIT
COL_SERVO    = (255, 120, 0)     # mavi    — servo HUD
COL_TEXT     = (255, 255, 255)
COL_BG       = (20, 20, 20)

# rt modulunun SAHI/esik sabitlerini bu testin degerleriyle ez
rt.CONF_THRESHOLD    = CONF_THRESHOLD
rt.SLICE_WIDTH       = SLICE_WIDTH
rt.SLICE_HEIGHT      = SLICE_HEIGHT
rt.OVERLAP_RATIO     = OVERLAP_RATIO
rt.NMS_IOU_THRESHOLD = NMS_IOU_THRESHOLD


def build_servo_config() -> ServoConfig:
    """Bu testin modul sabitlerinden yeniden kullanilabilir ServoConfig kurar."""
    return ServoConfig(
        lock_box_frac=LOCK_BOX_FRAC,
        lock_hyst_scale=LOCK_HYST_SCALE,
        lock_confirm_frames=LOCK_CONFIRM_FRAMES,
        lock_release_frames=LOCK_RELEASE_FRAMES,
        camera_fov_h_deg=CAMERA_FOV_H_DEG,
        kp_yaw=SERVO_KP_YAW, kd_yaw=SERVO_KD_YAW, kp_vert=SERVO_KP_VERT,
        pitch_max=SERVO_PITCH_MAX, yaw_max=SERVO_YAW_MAX, thr_max=SERVO_THR_MAX,
        err_tau_s=SERVO_ERR_TAU, ramp_s=SERVO_RAMP_S, slew=dict(SERVO_SLEW),
        lost_timeout_s=SERVO_LOST_TIMEOUT,
        yaw_sign=YAW_SIGN, throttle_sign=THROTTLE_SIGN,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  HIZLI CSRT (sort_csrt fabrikasini hafifletilmis parametrelerle ezer)
# ═══════════════════════════════════════════════════════════════════════════════
# Klonlanmis repoyu degistirmemek icin monkeypatch: CSRTTracker, tracker'i modul
# global'i _create_csrt_tracker uzerinden actigindan buradaki ezme yeterlidir.

import sort_csrt as _sort_mod                           # noqa: E402

_orig_create_csrt = _sort_mod._create_csrt_tracker


def _fast_csrt_tracker():
    """Hafifletilmis CSRT: olcek arama 33->9, ADMM 4->2, sablon 200->150,
    kanal agirliklari kapali. RTX 4050 uzerinde tek hedefte ~13.5 -> ~9 ms;
    kucuk/yavas buyuyen hedeflerde takip kalitesi kaybi ihmal edilebilir."""
    if not CSRT_FAST:
        return _orig_create_csrt()
    try:
        p = cv2.TrackerCSRT_Params()
        p.number_of_scales    = CSRT_NUM_SCALES
        p.admm_iterations     = CSRT_ADMM_ITERS
        p.template_size       = CSRT_TEMPLATE_SIZE
        p.use_channel_weights = False
        return cv2.TrackerCSRT_create(p)
    except (AttributeError, TypeError, cv2.error):
        return _orig_create_csrt()   # eski OpenCV: varsayilan fabrikaya don


_sort_mod._create_csrt_tracker = _fast_csrt_tracker


def prune_to_lock_target(sort_tracker: Sort, lock: "LockOnManager") -> int:
    """TAKIP fazina girerken kilit hedefi disindaki tracker'lari atar.

    Kilitlenme/servo tek hedefle calisir; her ek tracker her karede tam CSRT
    maliyeti ekler. Atilan tracker sayisini doner. Kilit hedefi yoksa merkeze
    en yakin tracker tutulur.
    """
    if len(sort_tracker.trackers) <= 1:
        return 0

    keep = None
    if lock.target_id is not None:
        for t in sort_tracker.trackers:
            if t.id + 1 == lock.target_id:      # SORT disariya id+1 verir
                keep = t
                break
    if keep is None:
        cx0, cy0 = lock.cx, lock.cy
        keep = min(
            sort_tracker.trackers,
            key=lambda t: ((t.bbox[0] + t.bbox[2]) / 2 - cx0) ** 2
                        + ((t.bbox[1] + t.bbox[3]) / 2 - cy0) ** 2,
        )
    dropped = len(sort_tracker.trackers) - 1
    sort_tracker.trackers = [keep]
    return dropped



# ═══════════════════════════════════════════════════════════════════════════════
#  SORT SESSIZ CAGRI + CIZIM YARDIMLARI
# ═══════════════════════════════════════════════════════════════════════════════

# Tek seferlik devnull — her cagride StringIO tahsis etmek yerine printler
# dogrudan cope yazilir (TAKIP fazinda her karede cagrildigi icin onemli).
_DEVNULL = open(os.devnull, "w")


def _quiet_sort_update(tracker: Sort, frame: np.ndarray, dets: np.ndarray) -> np.ndarray:
    """sort_csrt.py icindeki debug printlerini bastirarak Sort.update cagirir."""
    old_stdout = sys.stdout
    sys.stdout = _DEVNULL
    try:
        return tracker.update(frame, dets)
    finally:
        sys.stdout = old_stdout


def put_text_bg(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    scale: float = 0.6,
    fg: tuple = COL_TEXT,
    bg: tuple = COL_BG,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, 1)
    x, y = pos
    cv2.rectangle(img, (x - 2, y - th - bl - 2), (x + tw + 2, y + bl), bg, -1)
    cv2.putText(img, text, (x, y - bl), font, scale, fg, 1, cv2.LINE_AA)


def draw_lock_reticle(frame: np.ndarray, lock: LockOnManager) -> None:
    """Merkez kilitlenme dortgenini (kose braketli) ve durumunu cizer."""
    x1, y1, x2, y2 = lock.box
    if lock.locked:
        color, label = COL_LOCKED, "KILIT"
    elif lock.target_row is not None and lock._in_frames > 0:
        color, label = COL_ACQUIRE, "KILITLENIYOR"
    else:
        color, label = COL_SEARCH, "ARAMA"

    # Kose braketleri
    seg = max(10, (x2 - x1) // 5)
    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                             (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (px, py), (px + dx * seg, py), color, 2, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + dy * seg), color, 2, cv2.LINE_AA)

    # Merkez isareti
    cv2.drawMarker(frame, (lock.cx, lock.cy), color, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
    put_text_bg(frame, label, (x1, y1 - 8), scale=0.65, fg=color)

    # Kilitli hedefe cizgi + hedef kutusu vurgusu
    if lock.target_row is not None:
        row = lock.target_row
        tx, ty = int((row[0] + row[2]) / 2), int((row[1] + row[3]) / 2)
        bx1, by1, bx2, by2 = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        cv2.drawMarker(frame, (tx, ty), color, cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
        if lock.locked:
            cv2.line(frame, (lock.cx, lock.cy), (tx, ty), color, 1, cv2.LINE_AA)
        cls_id = int(row[5])
        name   = rt.CLASS_NAMES.get(cls_id, f"id:{cls_id}")
        put_text_bg(frame, f"ID:{int(row[6])} {name}", (bx1, by1 - 4), scale=0.5, fg=color)


def draw_servo_hud(
    frame: np.ndarray,
    active: bool,
    connected: bool,
    thr: float,
    pit: float,
    yaw: float,
    err_deg: tuple[float, float],
) -> None:
    h, w = frame.shape[:2]
    x = w - 275
    if not active:
        put_text_bg(frame, "SERVO: PASIF  [G]", (x, 30), scale=0.6, fg=(160, 160, 160))
        return
    put_text_bg(frame, "SERVO: AKTIF  [G]", (x, 30), scale=0.6, fg=COL_SERVO)
    put_text_bg(frame, f"err: {err_deg[0]:+5.1f} / {err_deg[1]:+5.1f} deg", (x, 58), scale=0.55, fg=COL_SERVO)
    put_text_bg(frame, f"yaw:{yaw:+.2f} pit:{pit:+.2f} thr:{thr:+.2f}", (x, 84), scale=0.55, fg=COL_SERVO)
    if not connected:
        put_text_bg(frame, "SDK BAGLI DEGIL!", (x, 110), scale=0.55, fg=COL_SEARCH)


def draw_hud(frame: np.ndarray, fps: float, phase: str, n_tracked: int, n_detect: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]
    lines = [
        (f"FPS: {fps:.1f}",     (10, 55),  0.9, (0, 255, 0)),
        (f"Takip: {n_tracked}", (10, 85),  0.7, (0, 255, 255)),
        (f"Tespit: {n_detect}", (10, 112), 0.7, (0, 200, 200)),
        (f"Faz: {phase}",       (10, 139), 0.7, (255, 255, 0)),
    ]
    for text, pos, scale, color in lines:
        cv2.putText(frame, text, pos, font, scale, color, 2, cv2.LINE_AA)
    put_text_bg(frame, "G: Servo | R: Sifirla | Q: Cikis", (w - 275, h - 15), scale=0.5)


def draw_timer_bar(frame: np.ndarray, elapsed: float, total: float, label: str, color: tuple) -> None:
    h, w = frame.shape[:2]
    bar_h  = 8
    fill_w = int(w * min(elapsed / max(total, 1e-6), 1.0))
    cv2.rectangle(frame, (0, 0), (w, bar_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, 0), (fill_w, bar_h), color, -1)
    put_text_bg(frame, f"{label} - kalan {max(0.0, total - elapsed):.1f}s",
                (10, bar_h + 18), scale=0.55, fg=color)


# ═══════════════════════════════════════════════════════════════════════════════
#  ANA DONGU
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── TRT engine ────────────────────────────────────────────────────────────
    inferencer = rt.TRTInferencer(ENGINE_PATH)
    inferencer.warmup(n=3)

    # ── Ekran yakalama ────────────────────────────────────────────────────────
    capture = rt.ScreenCapture(MONITOR_INDEX, RESIZE_SCALE)
    print("[INFO] Ilk kare bekleniyor...")
    while capture.get() is None:
        time.sleep(0.005)
    first = capture.get()
    frame_h, frame_w = first.shape[:2]
    print(f"[INFO] Kare boyutu: {frame_w}x{frame_h}")

    n_tiles = len(rt._get_slices(frame_h, frame_w))
    print(f"[INFO] SAHI: {SLICE_WIDTH}x{SLICE_HEIGHT} tile, overlap={OVERLAP_RATIO}, {n_tiles} tile/kare")

    # ── SORT-CSRT / kilit / servo ─────────────────────────────────────────────
    sort_tracker = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS,
                        iou_threshold=SORT_IOU_THRESH)
    servo_cfg = build_servo_config()
    lock  = LockOnManager(frame_w, frame_h, servo_cfg)
    servo = VisualServoController(frame_w, frame_h, servo_cfg)
    servo_active = False

    # ── SDK baglantisi (servo icin; basarisiz olsa da goruntu isleme calisir) ──
    sdk_connected = drone_sdk.connect()
    print(f"[INFO] Drone SDK: {'BAGLI' if sdk_connected else 'bagli degil (servo devre disi)'}")

    # ── Video yazici ──────────────────────────────────────────────────────────
    video_writer = None
    output_path  = "(kayit kapali)"
    if RECORD_VIDEO:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(OUTPUT_DIR / f"visual_servo_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        video_writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), OUTPUT_FPS, (frame_w, frame_h),
        )
        if not video_writer.isOpened():
            print("[WARN] Video yazici acilamadi; sadece ekranda gosterilecek.")
            video_writer = None
        else:
            print(f"[INFO] Video kaydi: {output_path}")

    win_name = "RF-DETR TRT SAHI + CSRT + Gorsel Servo"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    prev_time   = time.perf_counter()
    fps         = 0.0
    fps_alpha   = 0.9
    in_track    = False
    phase_start = time.perf_counter()
    frame_idx   = 0
    n_detect_current = 0
    thr = pit = yaw = 0.0
    prev_raw: Optional[np.ndarray] = None    # yinelenen kare tespiti (identity)
    track_tick   = 0                          # TAKIP fazi stride sayaci
    last_tracked = np.empty((0, 7))           # stride atlanan karede son sonuc

    print(f"\n{'='*64}\n"
          f"  Faz dongusu: TESPIT({DETECT_DURATION}s) -> TAKIP({TRACK_DURATION}s)\n"
          f"  G=Servo ac/kapa | R=Sifirla | Q=Cikis\n"
          f"{'='*64}\n")

    try:
        while True:
            frame = capture.get()
            # Yinelenen kare atlama: yakalama thread'i her karede YENI dizi
            # uretir; ayni dizi (identity) geldiyse yeni goruntu yok demektir.
            # CSRT'yi ayni kare uzerinde tekrar kosturmak bosa maliyettir.
            # Ayni sebeple frame.copy() gereksizdir: bu dizi thread tarafindan
            # bir daha yazilmaz, uzerine dogrudan annotasyon yapilabilir.
            if frame is None or frame is prev_raw:
                time.sleep(0.001)
                continue
            prev_raw = frame
            frame_idx += 1

            now = time.perf_counter()
            dt  = now - prev_time
            if dt > 0:
                cur_fps = 1.0 / dt
                fps = fps_alpha * fps + (1.0 - fps_alpha) * cur_fps if fps > 0 else cur_fps
            prev_time = now

            elapsed = now - phase_start
            tracked = np.empty((0, 7))
            raw_xyxy = np.empty((0, 4))

            # ══ TESPIT FAZI: SAHI + RF-DETR (TRT) + SORT-CSRT ══════════════════
            if not in_track:
                phase = "TESPIT"
                xyxy, confs, class_ids = rt.sahi_predict(inferencer, frame)
                raw_xyxy = xyxy
                n_detect_current = len(confs)

                if n_detect_current > 0:
                    dets = np.column_stack([xyxy, confs, class_ids])
                else:
                    dets = np.empty((0, 6))
                tracked = _quiet_sort_update(sort_tracker, frame, dets)

                draw_timer_bar(frame, elapsed, DETECT_DURATION, "TESPIT", COL_DETECT)

                if elapsed >= DETECT_DURATION:
                    if len(tracked) > 0:
                        # Kilit hedefi disindaki tracker'lari at: her ek tracker
                        # TAKIP fazinda her karede tam CSRT maliyeti ekler.
                        # lock.target_id onceki karelerin lock.update'inden guncel
                        if TRACK_ONLY_LOCK_TARGET:
                            dropped = prune_to_lock_target(sort_tracker, lock)
                            if dropped:
                                print(f"[KARE {frame_idx}] {dropped} yan hedef birakildi (hiz).")
                        print(f"[KARE {frame_idx}] TAKIP fazina gecildi — "
                              f"{len(sort_tracker.trackers)} hedef")
                        in_track = True
                        track_tick = 0
                        last_tracked = tracked
                    phase_start = time.perf_counter()

            # ══ TAKIP FAZI: sadece SORT-CSRT (yuksek FPS) ══════════════════════
            else:
                phase = "TAKIP"
                n_detect_current = 0

                # Stride: CSRT her TRACK_STRIDE karede bir kosulur; aradaki
                # karelerde son kutu yeniden kullanilir (hedef kareler arasi
                # birkac piksel oynar, kilit/servo icin fark ihmal edilebilir).
                track_tick += 1
                if track_tick % TRACK_STRIDE == 0 or len(last_tracked) == 0:
                    tracked = _quiet_sort_update(sort_tracker, frame, np.empty((0, 6)))
                    last_tracked = tracked
                    if len(tracked) == 0:
                        print(f"[KARE {frame_idx}] Hedefler kayboldu — TESPIT fazina donuluyor.")
                        in_track = False
                        phase_start = time.perf_counter()
                else:
                    tracked = last_tracked

                draw_timer_bar(frame, elapsed, TRACK_DURATION, "TAKIP", COL_ACQUIRE)

                if in_track and elapsed >= TRACK_DURATION:
                    in_track = False
                    phase_start = time.perf_counter()

            # ══ KILITLENME ═════════════════════════════════════════════════════
            target_row = lock.update(tracked)

            # ══ GORSEL SERVO (yalnizca aktifken komut gonderir) ════════════════
            if servo_active:
                if lock.locked and target_row is not None:
                    tx = (target_row[0] + target_row[2]) / 2.0
                    ty = (target_row[1] + target_row[3]) / 2.0
                    thr, pit, yaw, alive = servo.update((tx, ty), dt)
                else:
                    thr, pit, yaw, alive = servo.update(None, dt)

                if drone_sdk.is_connected():
                    drone_sdk.set_control_surfaces(thr, pit, 0.0, yaw, True)

                if not alive:
                    print("[INFO] Hedef kaybi zaman asimi — servo kapatildi.")
                    servo_active = False
                    servo.reset()
                    if drone_sdk.is_connected():
                        drone_sdk.set_control_surfaces(0.0, 0.0, 0.0, 0.0, True)
            else:
                thr = pit = yaw = 0.0

            # ══ CIZIM ══════════════════════════════════════════════════════════
            for box in raw_xyxy:   # ham tespitler (ince yesil)
                cv2.rectangle(frame, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), COL_DETECT, 1)
            draw_lock_reticle(frame, lock)
            draw_servo_hud(frame, servo_active, drone_sdk.is_connected(),
                           thr, pit, yaw, servo.filtered_error_deg)
            draw_hud(frame, fps, phase, len(tracked), n_detect_current)

            if video_writer:
                video_writer.write(frame)
            cv2.imshow(win_name, frame)

            # ══ TUSLAR ═════════════════════════════════════════════════════════
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("g"):
                if not servo_active:
                    if not drone_sdk.is_connected():
                        sdk_connected = drone_sdk.connect()
                    if drone_sdk.is_connected():
                        servo_active = True
                        servo.reset()
                        drone_sdk.set_arm(True)
                        print("[INFO] GORSEL SERVO AKTIF — kilitli hedefe yonelim basliyor.")
                    else:
                        print("[WARN] Simulator bagli degil; servo acilamadi.")
                else:
                    servo_active = False
                    servo.reset()
                    if drone_sdk.is_connected():
                        drone_sdk.set_control_surfaces(0.0, 0.0, 0.0, 0.0, True)
                    print("[INFO] Gorsel servo KAPATILDI — sadece goruntu isleme + kilit.")
            elif key == ord("r"):
                print("[INFO] Sifirlama — TESPIT fazina donuluyor.")
                in_track = False
                CSRTTracker.count = 0
                sort_tracker = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS,
                                    iou_threshold=SORT_IOU_THRESH)
                lock.reset()
                track_tick = 0
                last_tracked = np.empty((0, 7))
                phase_start = time.perf_counter()

    finally:
        if drone_sdk.is_connected():
            drone_sdk.set_control_surfaces(0.0, 0.0, 0.0, 0.0, False)
            drone_sdk.disconnect()
        capture.stop()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print(f"\n[INFO] Toplam {frame_idx} kare islendi.")
        print(f"[INFO] Video: {output_path}")


if __name__ == "__main__":
    main()
