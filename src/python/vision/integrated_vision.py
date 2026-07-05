#!/usr/bin/env python3
"""
Integrated Vision System (RF-DETR + TCTrack) Module
This module encapsulates all object detection, target tracking, exclusion zones,
and vision-based guidance command calculation into a single, clean class.
"""

import os
import sys
import time
import math
import cv2
import numpy as np
import torch
from pathlib import Path

# Resolve paths
VISION_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path(os.getcwd())
PROJECT_ROOT = VISION_DIR.parents[2]
TCTRACK_ROOT = VISION_DIR / "TCTrack"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TCTRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TCTRACK_ROOT))

# RF-DETR TensorRT Modules
from src.python.vision.rfdetr_inference.run_rfdetr_tensorrt import (
    TRTInferencer,
    sahi_predict,
    EXCLUSION_ZONES,
    apply_exclusion_zones,
    draw_exclusion_zones,
    CLASS_NAMES
)

# TCTrack Modules
from pysot.core.config import cfg
from pysot.models.utile_tctrack.model_builder import ModelBuilder_tctrack
from pysot.tracker.tctrack_tracker import TCTrackTracker
from pysot.utils.model_load import load_pretrain


class IntegratedVisionSystem:
    """
    Manages both detection (RF-DETR) and tracking (TCTrack) states.
    Produces steering commands to guide the UAV towards the target.
    """
    def __init__(self, config: dict, engine_path: str = None, tctrack_config: str = None, tctrack_snapshot: str = None):
        self.config = config
        
        # Model path settings
        self.engine_path = engine_path or str(PROJECT_ROOT / "object_detection" / "inference_model.engine")
        self.tctrack_config = tctrack_config or str(TCTRACK_ROOT / "experiments" / "TCTrack" / "config.yaml")
        self.tctrack_snapshot = tctrack_snapshot or str(TCTRACK_ROOT / "snapshot" / "general_model.pth")
        
        # Parse parameters from config, with default fallback values
        self.conf_threshold = float(config.get("vision", {}).get("conf_threshold", 0.50))
        self.track_score_min = float(config.get("vision", {}).get("track_score_min", 0.30))
        self.track_timeout = float(config.get("vision", {}).get("track_timeout", 4.0))
        self.kp_vision_yaw = float(config.get("vision", {}).get("kp_vision_yaw", 0.45))
        self.kp_vision_pitch = float(config.get("vision", {}).get("kp_vision_pitch", 0.25))
        self.vision_max_dist = float(config.get("vision", {}).get("vision_max_dist", 2500.0))
        
        # GPU Device settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize TRT Detector
        print(f"[INFO] TRT Engine yukleniyor: {self.engine_path}")
        self.inferencer = TRTInferencer(self.engine_path)
        
        # Initialize TCTrack Tracker
        print(f"[INFO] TCTrack config yukleniyor: {self.tctrack_config}")
        cfg.merge_from_file(self.tctrack_config)
        print(f"[INFO] TCTrack snapshot yukleniyor: {self.tctrack_snapshot}")
        self.tc_model = ModelBuilder_tctrack("test")
        self.tc_model = load_pretrain(self.tc_model, self.tctrack_snapshot).eval().to(self.device)
        self.tracker = TCTrackTracker(self.tc_model)
        self.hp = [cfg.TRACK.PENALTY_K, cfg.TRACK.WINDOW_INFLUENCE, cfg.TRACK.LR]
        
        # State variables
        self.tracker_enabled = True
        self.tracking = False
        self.track_start_time = 0.0
        self.tracked_bbox = None
        self.target_class_name = "None"
        self.tracker_score = 0.0
        self.detection_score = 0.0
        
        # Scaling
        self.resize_scale = float(config.get("vision", {}).get("resize_scale", 0.7))

    def update(self, frame: np.ndarray, dist_3d: float, dz: float, dt: float) -> dict:
        """
        Updates the tracking state machine and returns flight guidance commands if target is active.
        
        Args:
            frame (np.ndarray): Raw camera frame.
            dist_3d (float): 3D distance to the target (cm).
            dz (float): Relative altitude offset (cm).
            dt (float): Loop delta time (s).
            
        Returns:
            dict: Result state containing guidance commands, scores, and annotated frame.
        """
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        
        yaw_cmd = 0.0
        pitch_cmd = 0.0
        yaw_err = 0.0
        
        # 1. Check Distance Threshold
        if dist_3d > self.vision_max_dist:
            if self.tracking:
                print(f"[INFO] Hedef cok uzakta ({dist_3d:.0f}cm > {self.vision_max_dist}cm). Konum Kilavuzluguna donuluyor.")
            self.tracking = False
            self.tracked_bbox = None
            self.tracker_score = 0.0
        else:
            if not self.tracker_enabled:
                # DETECT-ONLY MODE: Run RF-DETR on every frame
                predict_frame = apply_exclusion_zones(frame)
                xyxy, confs, class_ids = sahi_predict(self.inferencer, predict_frame, conf_threshold=self.conf_threshold)
                
                self.tracking = False
                self.tracked_bbox = None
                self.tracker_score = 0.0
                
                if len(xyxy) > 0:
                    planes_idx = [i for i, cid in enumerate(class_ids) if int(cid) == 1]
                    if planes_idx:
                        best_idx = planes_idx[np.argmax(confs[planes_idx])]
                        box = xyxy[best_idx]
                        x1, y1, x2, y2 = box
                        bw, bh = x2 - x1, y2 - y1
                        
                        self.tracked_bbox = (float(x1), float(y1), float(bw), float(bh))
                        target_class_id = int(class_ids[best_idx])
                        self.target_class_name = CLASS_NAMES.get(target_class_id, f"id:{target_class_id}")
                        self.tracker_score = float(confs[best_idx])
                        self.detection_score = float(confs[best_idx])
                        self.tracking = True
                        
                        # Draw detection bounding box (green)
                        self.draw_target_bracket(display_frame, self.tracked_bbox, color=(0, 255, 0), thickness=2)
            else:
                # TCTRACK MODE: Detect once, then track frame-to-frame using TCTrack
                if not self.tracking:
                    predict_frame = apply_exclusion_zones(frame)
                    xyxy, confs, class_ids = sahi_predict(self.inferencer, predict_frame, conf_threshold=self.conf_threshold)
                    
                    if len(xyxy) > 0:
                        planes_idx = [i for i, cid in enumerate(class_ids) if int(cid) == 1]
                        if planes_idx:
                            best_idx = planes_idx[np.argmax(confs[planes_idx])]
                            
                            box = xyxy[best_idx]
                            x1, y1, x2, y2 = box
                            bw, bh = x2 - x1, y2 - y1
                            
                            self.tracked_bbox = (float(x1), float(y1), float(bw), float(bh))
                            target_class_id = int(class_ids[best_idx])
                            self.target_class_name = CLASS_NAMES.get(target_class_id, f"id:{target_class_id}")
                            self.tracker_score = float(confs[best_idx])
                            self.detection_score = float(confs[best_idx])
                            
                            # Initialize TCTrack tracker
                            self.tracker.init(frame, self.tracked_bbox)
                            self.tracking = True
                            self.track_start_time = time.perf_counter()
                            print(f"[INFO] HEDEF (PLANE) GORUS ALANINDA! Kamera Takibine Geciliyor (Conf: {self.tracker_score:.2f})")
                else:
                    # Update tracker
                    elapsed_time = time.perf_counter() - self.track_start_time
                    if elapsed_time >= self.track_timeout:
                        print(f"[INFO] Takip zaman asimi ({self.track_timeout}s). Konum Kilavuzluguna geri donuluyor.")
                        self.tracking = False
                    else:
                        tracked = self.tracker.track(frame, self.hp)
                        self.tracked_bbox = tuple(float(v) for v in tracked["bbox"])
                        self.tracker_score = float(tracked["best_score"])
                        
                        if self.tracker_score < self.track_score_min:
                            print(f"[INFO] Takip kalitesi dustu (Skor: {self.tracker_score:.2f}). Konum Kilavuzluguna geri donuluyor.")
                            self.tracking = False
                        else:
                            # Draw tracking bounding box (red)
                            self.draw_target_bracket(display_frame, self.tracked_bbox, color=(0, 0, 255), thickness=2)
                            
        # 2. Command calculation (Vision Guidance)
        if self.tracking and self.tracked_bbox is not None:
            bx, by, bw, bh = self.tracked_bbox
            cx = bx + bw / 2.0
            cy = by + bh / 2.0
            
            # Normalize pixel errors from center [-1, 1]
            err_x = (cx - (w / 2.0)) / (w / 2.0)
            err_y = (cy - (h / 2.0)) / (h / 2.0)
            
            yaw_cmd = self.clamp(err_x * self.kp_vision_yaw, -1.0, 1.0)
            pitch_cmd = self.clamp(-err_y * self.kp_vision_pitch, -1.0, 1.0)
            yaw_err = err_x * 180.0
            
        # Draw exclusion zones
        display_frame = draw_exclusion_zones(display_frame)
        
        return {
            "tracking": self.tracking,
            "yaw_cmd": yaw_cmd,
            "pitch_cmd": pitch_cmd,
            "yaw_err": yaw_err,
            "display_frame": display_frame,
            "target_class": self.target_class_name if self.tracking else "None",
            "tracker_score": self.tracker_score if self.tracking else 0.0,
            "detection_score": self.detection_score if self.tracking else 0.0,
            "tracker_enabled": self.tracker_enabled,
            "remaining_time": max(0.0, self.track_timeout - (time.perf_counter() - self.track_start_time)) if self.tracking else 0.0
        }

    @staticmethod
    def clamp(val, min_val, max_val):
        return max(min_val, min(max_val, val))

    @staticmethod
    def draw_target_bracket(img, bbox, color=(0, 0, 255), thickness=2):
        """Draws fighter-jet style bracket corners around bounding box."""
        x, y, w, h = map(int, bbox)
        length = min(15, w // 4, h // 4)
        
        # Top-Left
        cv2.line(img, (x, y), (x + length, y), color, thickness)
        cv2.line(img, (x, y), (x, y + length), color, thickness)
        
        # Top-Right
        cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
        cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)
        
        # Bottom-Left
        cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
        cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)
        
        # Bottom-Right
        cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
        cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)

    def draw_hud(self, frame, mode, remaining_time=0.0, target_class="None", tracker_score=0.0,
                 fps=0.0, dist=0.0, dz=0.0, yaw_err=0.0, pitch_cmd=0.0, yaw_cmd=0.0, throttle=0.0,
                 detection_score=0.0):
        """Draws telemetry and control output panel onto the frame."""
        # HUD Panel Background
        cv2.rectangle(frame, (10, 10), (320, 230), (15, 15, 15), -1)
        cv2.rectangle(frame, (10, 10), (320, 230), (80, 80, 80), 1)

        # Active Guidance Mode status text
        if mode == "VISION":
            status_text = "GUIDANCE: VISION"
            status_color = (0, 0, 255) # Red
        else:
            status_text = "GUIDANCE: POSITION [EKF/GPS]"
            status_color = (0, 255, 255) # Yellow

        cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Target Class: {target_class}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Tracking status lines
        if mode == "VISION":
            if self.tracker_enabled:
                cv2.putText(frame, f"Track Time: {remaining_time:.1f}s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Det/Trk Score: {detection_score:.2f}/{tracker_score:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Track Time: DETECT-ONLY", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Det Conf: {tracker_score:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Track Time: --", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
            cv2.putText(frame, "Det/Trk Score: --", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

        # Telemetry variables
        cv2.line(frame, (20, 120), (310, 120), (50, 50, 50), 1)
        cv2.putText(frame, f"Dist 3D: {dist:.0f}/{self.vision_max_dist:.0f} cm", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Alt Diff (dz): {dz:.1f} cm", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Yaw Error: {yaw_err:.1f} deg", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Flight controller commands
        cv2.line(frame, (20, 180), (310, 180), (50, 50, 50), 1)
        cv2.putText(frame, f"CMD - Pitch: {pitch_cmd:+.2f} | Yaw: {yaw_cmd:+.2f}", (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, f"CMD - Throttle: {throttle:+.2f}", (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1, cv2.LINE_AA)
