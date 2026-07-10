"""
================================================================================
TEST: Rota Takibi -> Gorsel Servolama Devri (GPS + Goruntu Isleme Fuzyonu)
================================================================================
Dosya: tests/python/test_route_vision_handover.py

test_route_intercept.py (GPS rota takibi) + test_rfdetr_trt_visual_servo.py
(TRT+SAHI+CSRT gorsel kilit) birlesimi:

  FAZ 1 — ROTA (GPS):
    Dron, ogrenilen elips rotada hedefi GPS uzerinden takip eder
    (OrbitLockTracker + RouteFollowGuidance + kanitlanmis PID'ler).

  FAZ 2 — DEVIR (harmanlama):
    GPS mesafesi HANDOVER_DIST_CM altina indiginde VE gorsel pipeline hedefe
    kilitliyse (merkez dortgende >= VISION_MIN_LOCK_S), komutlar BLEND_S
    saniyede smoothstep ile GPS'ten gorsel servoya harmanlanir — ani stick
    sicramasi olmaz.

  FAZ 3 — GORSEL:
    Yaw/throttle/pitch tamamen goruntu hatasindan (VisualServoController:
    LPF + PD + slew + devreye-giris rampasi). GPS guidance arka planda
    calismaya DEVAM eder — geri dusus her an h