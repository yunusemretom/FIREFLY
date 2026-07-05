#!/usr/bin/env python3
"""
RF-DETR TensorRT + TCTrack Entegre Takip Test Scripti
Açıklama:
    1) RF-DETR TensorRT engine ile ekranda hedef tespit edilir.
    2) Tespit edilen hedef ile TCTrack tracker başlatılır.
    3) 4 saniye boyunca yalnızca TCTrack ile yüksek hızlı ve düşük gecikmeli takip sürdürülür.
    4) 4 saniye sonunda veya takip kaybedildiğinde tekrar RF-DETR tespit moduna dönülür.
    5) Yasaklı bölge (exclusion zones) maskelemesi tespit aşamasında etkindir.
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import mss
import torch

# Proje kök dizinini sys.path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# TCTrack kök dizinini sys.path'e ekle
TCTRACK_ROOT = PROJECT_ROOT / "src" / "python" / "vision" / "TCTrack"
if str(TCTRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TCTRACK_ROOT))

# RF-DETR TensorRT Modüllerini import et
from src.python.vision.rfdetr_inference.run_rfdetr_tensorrt import (
    TRTInferencer,
    sahi_predict,
    EXCLUSION_ZONES,
    SHOW_EXCLUSION_ZONES,
    apply_exclusion_zones,
    draw_exclusion_zones,
    CLASS_NAMES
)

# TCTrack modüllerini import et
from pysot.core.config import cfg
from pysot.models.utile_tctrack.model_builder import ModelBuilder_tctrack
from pysot.tracker.tctrack_tracker import TCTrackTracker
from pysot.utils.model_load import load_pretrain

# ─── Yapılandırma ────────────────────────────────────────────────────────────
ENGINE_PATH     = PROJECT_ROOT / "object_detection" / "inference_model.engine"
TCTRACK_CONFIG  = TCTRACK_ROOT / "experiments" / "TCTrack" / "config.yaml"
TCTRACK_SNAPSHOT= TCTRACK_ROOT / "snapshot" / "general_model.pth"

CONF_THRESHOLD  = 0.50          # RF-DETR güven eşiği
RESIZE_SCALE    = 0.4           # Ekran yakalama küçültme oranı
MONITOR_INDEX   = 1             # Yakalanacak ekran (1 = ana ekran)
OUTPUT_FPS      = 50.0          # Çıktı video FPS değeri
TRACK_TIMEOUT   = 4.0           # Algılamadan sonra TCTrack ile takip süresi (saniye)
TRACK_SCORE_MIN = 0.15          # Takip kayıp eşiği

OUTPUT_DIR      = Path("/home/tom/Documents/Projeler/DogFight")
OUTPUT_PATH     = OUTPUT_DIR / f"ekran_takip_tctrack_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

# ─── Yardımcı Görselleştirme Fonksiyonları ────────────────────────────────────
def draw_target_bracket(img, bbox, color=(0, 0, 255), thickness=2):
    """Fighter-jet stili köşeli hedef kilitleme çerçevesi çizer."""
    x, y, w, h = map(int, bbox)
    length = min(15, w // 4, h // 4)
    length = max(5, length)
    
    # Sol Üst
    cv2.line(img, (x, y), (x + length, y), color, thickness)
    cv2.line(img, (x, y), (x, y + length), color, thickness)
    # Sağ Üst
    cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)
    # Sol Alt
    cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)
    # Sağ Alt
    cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)
    
    # Merkez nokta
    cx, cy = x + w // 2, y + h // 2
    cv2.circle(img, (cx, cy), 2, color, -1)


def draw_hud(frame, tracking, remaining_time=0.0, target_class="None", tracker_score=0.0, fps=0.0):
    """Ekranın sol üst köşesine şık bir HUD bilgi ekranı çizer."""
    # HUD Arka plan paneli
    cv2.rectangle(frame, (10, 10), (280, 140), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, 10), (280, 140), (100, 100, 100), 1)

    # Durum bilgisi
    if tracking:
        status_text = "STATUS: LOCKED [TCTrack]"
        status_color = (0, 0, 255) # Kırmızı (Takip/Kilit aktif)
    else:
        status_text = "STATUS: SEARCHING [DETR]"
        status_color = (0, 255, 255) # Sarı (Arama aktif)

    cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Target Class: {target_class}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    if tracking:
        cv2.putText(frame, f"Track Time: {remaining_time:.1f}s", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Track Score: {tracker_score:.2f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Track Time: --", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)


# ─── Ana Program ──────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. RF-DETR Modelini Yükle
    inferencer = TRTInferencer(str(ENGINE_PATH))

    # 2. TCTrack Modelini Yükle
    print(f"[INFO] TCTrack config yükleniyor: {TCTRACK_CONFIG}")
    cfg.merge_from_file(str(TCTRACK_CONFIG))
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if cfg.CUDA else "cpu")

    print(f"[INFO] TCTrack snapshot yükleniyor: {TCTRACK_SNAPSHOT}")
    tc_model = ModelBuilder_tctrack("test")
    tc_model = load_pretrain(tc_model, str(TCTRACK_SNAPSHOT)).eval().to(device)
    tracker = TCTrackTracker(tc_model)
    hp = [cfg.TRACK.PENALTY_K, cfg.TRACK.WINDOW_INFLUENCE, cfg.TRACK.LR]

    print("[INFO] Entegre Pipeline Hazır.")

    # Ekran Yakalama Başlat
    video_writer = None
    prev_time = time.perf_counter()
    fps = 0.0
    fps_alpha = 0.9

    # Takip Durum Değişkenleri
    tracking = False
    track_start_time = 0.0
    tracked_bbox = None  # [x, y, w, h] formatında
    target_class_name = "None"
    tracker_score = 0.0

    with mss.MSS() as sct:
        if MONITOR_INDEX >= len(sct.monitors):
            raise RuntimeError(f"Geçersiz monitör indeksi: {MONITOR_INDEX}")

        monitor = sct.monitors[MONITOR_INDEX]
        print(f"[INFO] Ekran Yakalanıyor: {monitor}")
        print(f"[INFO] Video Kayıt Yolu: {OUTPUT_PATH}")
        print("[INFO] Çıkış için pencere üzerindeyken 'q' tuşuna basın.")

        while True:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            if RESIZE_SCALE != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)

            h, w = frame.shape[:2]

            # FPS Hesabı
            now = time.perf_counter()
            dt = now - prev_time
            if dt > 0:
                cur_fps = 1.0 / dt
                fps = (fps_alpha * fps + (1.0 - fps_alpha) * cur_fps) if fps > 0 else cur_fps
            prev_time = now

            # Görselleştirme için orijinal görüntüyü kopyala
            display_frame = frame.copy()

            if not tracking:
                # ─── RF-DETR TESPİT AŞAMASI ───────────────────────────────────
                # Yasaklı bölgeleri maskele
                predict_frame = apply_exclusion_zones(frame)
                
                # RF-DETR ile tahmin yap
                xyxy, confs, class_ids = sahi_predict(inferencer, predict_frame)

                # Eğer herhangi bir nesne tespit edildiyse
                if len(xyxy) > 0:
                    # Öncelikle sınıfı 1 (Plane) olan nesneyi seçmeye çalış, yoksa en yüksek güvenli olanı seç
                    best_idx = None
                    planes_idx = [i for i, cid in enumerate(class_ids) if int(cid) == 1]
                    
                    if planes_idx:
                        # En yüksek güvenli uçağı seç
                        best_idx = planes_idx[np.argmax(confs[planes_idx])]
                    else:
                        # Diğer sınıflardan en yüksek güvenli olanı seç
                        best_idx = np.argmax(confs)

                    box = xyxy[best_idx]
                    x1, y1, x2, y2 = box
                    bw, bh = x2 - x1, y2 - y1
                    
                    # TCTrack [x, y, w, h] formatında bbox bekler
                    tracked_bbox = (float(x1), float(y1), float(bw), float(bh))
                    target_class_id = int(class_ids[best_idx])
                    target_class_name = CLASS_NAMES.get(target_class_id, f"id:{target_class_id}")
                    tracker_score = float(confs[best_idx])

                    # TCTrack başlat
                    tracker.init(frame, tracked_bbox)
                    tracking = True
                    track_start_time = time.perf_counter()
                    print(f"[INFO] Hedef Algılandı ({target_class_name}, Conf: {tracker_score:.2f}). TCTrack başlatılıyor...")

            else:
                # ─── TCTRACK TAKİP AŞAMASI ────────────────────────────────────
                elapsed_time = time.perf_counter() - track_start_time
                remaining_time = max(0.0, TRACK_TIMEOUT - elapsed_time)

                if elapsed_time >= TRACK_TIMEOUT:
                    print(f"[INFO] {TRACK_TIMEOUT} saniyelik takip süresi doldu. Tekrar tespit moduna geçiliyor.")
                    tracking = False
                else:
                    # Takip güncelle
                    tracked = tracker.track(frame, hp)
                    tracked_bbox = tuple(float(v) for v in tracked["bbox"])
                    tracker_score = float(tracked["best_score"])

                    if tracker_score < TRACK_SCORE_MIN:
                        print(f"[INFO] Takip kalitesi düştü (Skor: {tracker_score:.2f}). Yeniden tespit deneniyor...")
                        tracking = False
                    else:
                        # Başarılı takip durumunda hedef kutusunu çiz
                        draw_target_bracket(display_frame, tracked_bbox, color=(0, 0, 255), thickness=2)

            # Yasaklı bölgeleri ekranda çiz (ayarlanmışsa)
            display_frame = draw_exclusion_zones(display_frame)

            # HUD Göstergesini çiz
            remaining_time = max(0.0, TRACK_TIMEOUT - (time.perf_counter() - track_start_time)) if tracking else 0.0
            draw_hud(display_frame, tracking, remaining_time, target_class_name, tracker_score, fps)

            # Video Kaydet
            if video_writer is None:
                oh, ow = display_frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(OUTPUT_PATH),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    OUTPUT_FPS,
                    (ow, oh),
                )
                if not video_writer.isOpened():
                    raise RuntimeError(f"Video kaydı başlatılamadı: {OUTPUT_PATH}")

            video_writer.write(display_frame)

            # Ekrana bas
            cv2.imshow("RF-DETR + TCTrack Entegre Takip Testi", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Kayıt tamamlandı: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
