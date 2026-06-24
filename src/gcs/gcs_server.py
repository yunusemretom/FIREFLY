"""
================================================================================
FIREFLY GCS — WebSocket Backend Server
================================================================================
Web tabanlı Yer Kontrol İstasyonu için Python WebSocket sunucusu.
drone_sdk.py ile doğrudan entegre çalışır.

Kullanım:
    python src/gcs/gcs_server.py

Tarayıcı bağlantısı:
    http://localhost:8080  (web arayüzü)
    ws://localhost:8765    (WebSocket telemetri/kontrol)
================================================================================
"""

import asyncio
import json
import logging
import math
import os
import sys
import threading
import time
import http.server
import socketserver
import functools
from pathlib import Path

# WebSocket kütüphanesi: pip install websockets
try:
    import websockets
except ImportError:
    print("[GCS] HATA: 'websockets' paketi bulunamadı.")
    print("[GCS] Kurmak için: pip install websockets")
    sys.exit(1)

# Proje kök dizinini Python path'e ekle (drone_sdk.py için)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import drone_sdk as drone
    SDK_AVAILABLE = True
    print(f"[GCS] drone_sdk.py yüklendi: {PROJECT_ROOT / 'drone_sdk.py'}")
except ImportError:
    SDK_AVAILABLE = False
    print("[GCS] UYARI: drone_sdk.py bulunamadı — Demo/Simülasyon modunda çalışılıyor.")

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[GCS %(levelname)s] %(message)s"
)
log = logging.getLogger("gcs_server")

# ─── Sabitler ─────────────────────────────────────────────────────────────────
WS_HOST = "localhost"
WS_PORT = 8765
HTTP_PORT = 8080
TELEMETRY_HZ = 20  # Saniyede kaç kez telemetri gönderilsin

WEB_DIR = Path(__file__).parent / "web"

# ─── Global Durum ─────────────────────────────────────────────────────────────
connected_clients: set = set()
current_settings = {
    "kp_yaw": 0.07,
    "kp_pitch": 0.035,
    "kp_thr": 0.01,
    "max_throttle": 1.0,
    "camera_url": "",
    "sim_host": "127.0.0.1",
    "sim_port": 12345,
}
auto_mode = False
arm_state = False

# Demo modu için sinüs dalgası telemetri
_demo_t = 0.0


def get_demo_telemetry() -> dict:
    """SDK yokken gerçekçi demo telemetri üretir."""
    global _demo_t
    _demo_t += 1.0 / TELEMETRY_HZ
    t = _demo_t

    return {
        "connected": False,
        "demo": True,
        "drone": {
            "position": [
                round(math.sin(t * 0.3) * 500, 2),
                round(math.cos(t * 0.2) * 300, 2),
                round(200 + math.sin(t * 0.5) * 50, 2),
            ],
            "rotation": [
                round(math.sin(t * 0.7) * 8, 2),   # Roll
                round(math.sin(t * 0.4) * 12, 2),  # Pitch
                round((t * 15) % 360, 2),           # Yaw
            ],
            "velocity": [
                round(math.sin(t) * 100, 2),
                round(math.cos(t) * 80, 2),
                round(math.sin(t * 1.5) * 30, 2),
            ],
            "speed": round(abs(math.sin(t * 0.6)) * 1200 + 400, 2),
            "altitude": round(200 + math.sin(t * 0.5) * 50, 2),
        },
        "target": {
            "position": [
                round(math.cos(t * 0.15) * 800, 2),
                round(math.sin(t * 0.12) * 600, 2),
                round(300 + math.sin(t * 0.3) * 100, 2),
            ],
            "rotation": [0.0, 0.0, round((t * 20) % 360, 2)],
            "speed": 1500.0,
        },
        "controls": {
            "throttle": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "yaw": 0.0,
            "arm": False,
            "auto_mode": False,
        },
    }


def get_live_telemetry() -> dict:
    """drone_sdk.py'den gerçek telemetri alır."""
    t = drone.get_telemetry()
    d = t["drone"]
    tgt = t["target"]
    return {
        "connected": drone.is_connected(),
        "demo": False,
        "drone": {
            "position": list(d["position"]),
            "rotation": list(d["rotation"]),
            "velocity": list(d["velocity"]),
            "speed": round(d["speed"], 2),
            "altitude": round(d["altitude"], 2),
        },
        "target": {
            "position": list(tgt["position"]),
            "rotation": list(tgt["rotation"]),
            "speed": round(tgt["speed"], 2),
        },
        "controls": {
            "throttle": drone._drone.throttle,
            "pitch": drone._drone.pitch,
            "roll": drone._drone.roll,
            "yaw": drone._drone.yaw,
            "arm": drone._drone.arm,
            "auto_mode": auto_mode,
        },
    }


async def telemetry_broadcaster():
    """Her istemciye periyodik telemetri gönderir."""
    global connected_clients
    interval = 1.0 / TELEMETRY_HZ
    while True:
        if connected_clients:
            try:
                if SDK_AVAILABLE and drone.is_connected():
                    data = get_live_telemetry()
                else:
                    data = get_demo_telemetry()

                payload = json.dumps({"type": "telemetry", "data": data})
                dead = set()
                for client_ws in list(connected_clients):
                    try:
                        await client_ws.send(payload)
                    except Exception:
                        dead.add(client_ws)
                connected_clients -= dead
            except Exception as e:
                log.warning(f"Telemetri broadcast hatası: {e}")
        await asyncio.sleep(interval)


async def handle_client(websocket):
    """Gelen WebSocket bağlantısını yönetir."""
    global auto_mode, arm_state
    connected_clients.add(websocket)
    client_addr = websocket.remote_address
    log.info(f"İstemci bağlandı: {client_addr}")

    # Bağlantı anında ayarları gönder
    await websocket.send(json.dumps({
        "type": "settings",
        "data": current_settings
    }))

    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
                mtype = msg.get("type", "")

                if mtype == "connect_drone":
                    if SDK_AVAILABLE:
                        host = msg.get("host", "127.0.0.1")
                        port = int(msg.get("port", 12345))
                        ok = drone.connect(host=host, port=port)
                        await websocket.send(json.dumps({
                            "type": "connect_result",
                            "success": ok,
                            "message": "Bağlandı" if ok else "Bağlantı başarısız"
                        }))
                    else:
                        await websocket.send(json.dumps({
                            "type": "connect_result",
                            "success": False,
                            "message": "Demo mod — SDK mevcut değil"
                        }))

                elif mtype == "disconnect_drone":
                    if SDK_AVAILABLE:
                        drone.disconnect()
                    await websocket.send(json.dumps({
                        "type": "connect_result",
                        "success": False,
                        "message": "Bağlantı kesildi"
                    }))

                elif mtype == "control":
                    if SDK_AVAILABLE and drone.is_connected():
                        throttle = float(msg.get("throttle", 0.0))
                        pitch = float(msg.get("pitch", 0.0))
                        roll = float(msg.get("roll", 0.0))
                        yaw = float(msg.get("yaw", 0.0))
                        arm = bool(msg.get("arm", False))
                        drone.set_control_surfaces(throttle, pitch, roll, yaw, arm)
                        arm_state = arm

                elif mtype == "set_arm":
                    arm_state = bool(msg.get("value", False))
                    if SDK_AVAILABLE and drone.is_connected():
                        drone.set_arm(arm_state)

                elif mtype == "set_auto_mode":
                    auto_mode = bool(msg.get("value", False))
                    log.info(f"Otonom mod: {'AKTİF' if auto_mode else 'PASİF'}")

                elif mtype == "update_settings":
                    settings_data = msg.get("data", {})
                    current_settings.update(settings_data)
                    log.info(f"Ayarlar güncellendi: {settings_data}")
                    await websocket.send(json.dumps({
                        "type": "settings_saved",
                        "success": True
                    }))

                elif mtype == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))

            except json.JSONDecodeError:
                log.warning(f"Geçersiz JSON: {raw[:100]}")
            except Exception as e:
                log.error(f"Mesaj işleme hatası: {e}")

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        log.info(f"İstemci ayrıldı: {client_addr}")


class SilentHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """Web dosyalarını sunar, loglama yapmaz."""
    def log_message(self, format, *args):
        pass  # Sessiz

    def end_headers(self):
        # CORS headers
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def run_http_server():
    """Statik web dosyalarını HTTP üzerinden sunar."""
    if not WEB_DIR.exists():
        log.warning(f"Web dizini bulunamadı: {WEB_DIR}")
        return

    handler = functools.partial(SilentHTTPHandler, directory=str(WEB_DIR))
    with socketserver.TCPServer(("", HTTP_PORT), handler) as httpd:
        httpd.allow_reuse_address = True
        log.info(f"HTTP sunucusu başlatıldı: http://localhost:{HTTP_PORT}")
        httpd.serve_forever()


async def main():
    log.info("=" * 60)
    log.info("  🔥 FIREFLY GCS — WebSocket Backend")
    log.info("=" * 60)
    log.info(f"  SDK Durumu: {'✅ Hazır' if SDK_AVAILABLE else '⚠️  Demo Modu'}")
    log.info(f"  WebSocket:  ws://{WS_HOST}:{WS_PORT}")
    log.info(f"  Web Arayüzü: http://localhost:{HTTP_PORT}")
    log.info("=" * 60)

    # HTTP sunucusunu ayrı thread'de çalıştır
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()

    # WebSocket sunucusunu başlat
    async with websockets.serve(handle_client, WS_HOST, WS_PORT):
        await asyncio.gather(
            telemetry_broadcaster(),
            asyncio.Future()  # Sonsuza kadar çalış
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("GCS sunucusu kapatılıyor...")
        if SDK_AVAILABLE:
            drone.disconnect()
