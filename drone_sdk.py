"""
================================================================================
DRONE OF WAR - COMPETITION SDK (v2.1)
================================================================================
Bu dosya, Unreal Engine tabanlı Drone Of War oyunu ile Python arasındaki 
TCP iletişimini sağlayan resmi SDK dosyasıdır. 

YARIŞMACI NOTU:
Sadece bu dosyayı projenize import ederek dronunuzu kontrol edebilirsiniz.
Kendi AI mantığınızı, bu SDK'daki 'get_' fonksiyonları ile telemetri alıp,
'set_' fonksiyonları ile dronu yönlendirerek kurmalısınız.

Örnek Kullanım:
    import drone_sdk as drone
    drone.connect()
    drone.set_arm(True)
    pos = drone.get_drone_location()
================================================================================
"""

import socket
import threading
import time
import logging

# Log seviyesini sadece hataları gösterecek şekilde ayarlıyoruz.
logging.basicConfig(level=logging.ERROR, format='[DroneSDK] %(levelname)s: %(message)s')

class _DroneInternal:
    """
    Dahili iletişim sınıfı. Yarışmacıların bu sınıfı direkt kullanması gerekmez.
    Public API fonksiyonları bu sınıfın global örneği (_drone) üzerinden çalışır.
    """
    def __init__(self):
        self.host = '127.0.0.1'
        self.port = 12345
        self.sock = None
        self.is_connected = False
        self.lock = threading.Lock()
        
        # Anlık Kontrol Değerleri
        self.throttle = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.arm = False
        
        # Telemetri Veri Yapısı
        self.telemetry = {
            "drone": {
                "position": (0.0, 0.0, 0.0), # X, Y, Z
                "rotation": (0.0, 0.0, 0.0), # Roll, Pitch, Yaw
                "velocity": (0.0, 0.0, 0.0), # X, Y, Z hız vektörü
                "speed": 0.0,                # cm/s cinsinden toplam hız
                "altitude": 0.0              # Z eksenindeki yükseklik
            },
            "target": {
                "position": (0.0, 0.0, 0.0), # X, Y, Z
                "rotation": (0.0, 0.0, 0.0), # Roll, Pitch, Yaw
                "speed": 0.0                 # cm/s cinsinden hız
            }
        }
        
        self._stop_event = threading.Event()
        self._receive_thread = None

    def connect(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.host, self.port))
            self.is_connected = True
            self._stop_event.clear()
            self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._receive_thread.start()
            return True
        except Exception as e:
            return False

    def disconnect(self):
        self._stop_event.set()
        if self.sock: self.sock.close()
        self.is_connected = False

    def send_inputs(self):
        if not self.is_connected: return
        arm_val = 1 if self.arm else 0
        # Format: throttle,pitch,roll,yaw,arm\n
        msg = f"{self.throttle:.4f},{self.pitch:.4f},{self.roll:.4f},{self.yaw:.4f},{arm_val}\n"
        try:
            self.sock.sendall(msg.encode('utf-8'))
        except:
            self.is_connected = False

    def _receive_loop(self):
        buffer = ""
        while not self._stop_event.is_set():
            try:
                data = self.sock.recv(2048).decode('utf-8')
                if not data: break
                buffer += data
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line: self._parse_telemetry(line)
            except: break
        self.is_connected = False

    def _parse_telemetry(self, line):
        try:
            v = [float(x) for x in line.split(",")]
            # Format: Drone(0-10), Target(11-17)
            if len(v) >= 18:
                with self.lock:
                    self.telemetry["drone"]["position"] = (v[0], v[1], v[2])
                    self.telemetry["drone"]["rotation"] = (v[3], v[4], v[5])
                    self.telemetry["drone"]["velocity"] = (v[6], v[7], v[8])
                    self.telemetry["drone"]["speed"] = v[9]
                    self.telemetry["drone"]["altitude"] = v[10]
                    self.telemetry["target"]["position"] = (v[11], v[12], v[13])
                    self.telemetry["target"]["rotation"] = (v[14], v[15], v[16])
                    self.telemetry["target"]["speed"] = v[17]
        except: pass

# Global SDK Örneği
_drone = _DroneInternal()

# --- PUBLIC API (YARIŞMACI FONKSİYONLARI) ---

def connect(host='127.0.0.1', port=12345):
    """Oyuna bağlantı kurar. (Varsayılan: localhost:12345)"""
    return _drone.connect(host, port)

def disconnect():
    """Bağlantıyı güvenli bir şekilde kapatır."""
    _drone.disconnect()

def is_connected():
    """Bağlantı durumunu döner (True/False)."""
    return _drone.is_connected

# --- KONTROL FONKSİYONLARI ---

def set_throttle(value: float):
    """Dronun gaz (yükselme/alçalma) değerini ayarlar. Aralığı: [0.0 - 1.0]"""
    _drone.throttle = max(0.0, min(1.0, value))
    _drone.send_inputs()

def set_pitch(value: float):
    """Dronun ileri/geri eğilme değerini ayarlar. Aralığı: [-1.0 - 1.0]"""
    _drone.pitch = max(-1.0, min(1.0, value))
    _drone.send_inputs()

def set_roll(value: float):
    """Dronun sağa/sola yatış değerini ayarlar. Aralığı: [-1.0 - 1.0]"""
    _drone.roll = max(-1.0, min(1.0, value))
    _drone.send_inputs()

def set_yaw(value: float):
    """Dronun kendi ekseninde dönme değerini ayarlar. Aralığı: [-1.0 - 1.0]"""
    _drone.yaw = max(-1.0, min(1.0, value))
    _drone.send_inputs()

def set_arm(state: bool):
    """Dronu aktif (True) veya pasif (False) hale getirir."""
    _drone.arm = state
    _drone.send_inputs()

def set_control_surfaces(throttle: float, pitch: float, roll: float, yaw: float, arm: bool):
    """Tüm kontrol yüzeylerini tek TCP satırında gönderir (ara karelerde throttle/pitch uyumsuzluğunu önler)."""
    _drone.throttle = max(0.0, min(1.0, throttle))
    _drone.pitch = max(-1.0, min(1.0, pitch))
    _drone.roll = max(-1.0, min(1.0, roll))
    _drone.yaw = max(-1.0, min(1.0, yaw))
    _drone.arm = arm
    _drone.send_inputs()

# --- TELEMETRİ FONKSİYONLARI ---

def get_drone_location():
    """Dronun anlık konumunu döner: (X, Y, Z)"""
    with _drone.lock: return _drone.telemetry["drone"]["position"]

def get_drone_rotation():
    """Dronun anlık rotasyonunu döner: (Roll, Pitch, Yaw)"""
    with _drone.lock: return _drone.telemetry["drone"]["rotation"]

def get_drone_speed():
    """Dronun anlık toplam hızını (cm/s) döner."""
    with _drone.lock: return _drone.telemetry["drone"]["speed"]

def get_drone_altitude():
    """Dronun anlık irtifasını (Z ekseni) döner."""
    with _drone.lock: return _drone.telemetry["drone"]["altitude"]

def get_target_location():
    """Hedefin anlık konumunu döner: (X, Y, Z)"""
    with _drone.lock: return _drone.telemetry["target"]["position"]

def get_target_rotation():
    """Hedefin anlık rotasyonunu döner: (Roll, Pitch, Yaw)"""
    with _drone.lock: return _drone.telemetry["target"]["rotation"]

def get_target_speed():
    """Hedefin anlık hızını (cm/s) döner."""
    with _drone.lock: return _drone.telemetry["target"]["speed"]

def get_telemetry():
    """Tüm telemetri verilerini bir 'dictionary' olarak döner."""
    with _drone.lock: return _drone.telemetry.copy()
