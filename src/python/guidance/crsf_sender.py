"""
CRSF (TBS Crossfire) packet generator and serial transmitter for flight controller.

CRSF frame formati (TBS Crossfire, RC_CHANNELS_PACKED = 0x16):
  [0]      Sync byte           = 0xC8  (Flight Controller adresi)
  [1]      Frame length        = payload_len + 2  (type + payload + crc)
  [2]      Type                = 0x16
  [3..24]  Payload             = 16 kanal x 11 bit, little-endian paketli (22 byte)
  [25]     CRC8                = poly 0xD5, [type..payload] uzerinden

Kanal degerleri 11-bit (0-2047), standart CRSF araligi:
  172  = min  (~988us)
  992  = orta (~1500us)
  1811 = max  (~2012us)
"""

import struct

try:
    import serial  # pyserial
except ImportError:
    serial = None


CRSF_SYNC = 0xC8
CRSF_TYPE_RC_CHANNELS_PACKED = 0x16
CRSF_NUM_CHANNELS = 16
CRSF_CHANNEL_MIN = 172
CRSF_CHANNEL_MID = 992
CRSF_CHANNEL_MAX = 1811

def _make_crc8_table(poly=0xD5):
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
        table.append(crc)
    return table


_CRC8_TABLE = _make_crc8_table()


def _crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = _CRC8_TABLE[crc ^ b]
    return crc


def _normalize_to_crsf(value: float) -> int:
    value = max(-1.0, min(1.0, float(value)))
    span = (CRSF_CHANNEL_MAX - CRSF_CHANNEL_MIN) / 2.0  # ~819.5
    ch = int(round(CRSF_CHANNEL_MID + value * span))
    return max(CRSF_CHANNEL_MIN, min(CRSF_CHANNEL_MAX, ch))


def pack_rc_channels(channels_11bit) -> bytes:
    if len(channels_11bit) != CRSF_NUM_CHANNELS:
        raise ValueError(f"CRSF_NUM_CHANNELS={CRSF_NUM_CHANNELS} kanal gerekli, {len(channels_11bit)} geldi")

    bit_buffer = 0
    bit_count = 0
    out = bytearray()
    for ch in channels_11bit:
        ch = max(0, min(2047, int(ch)))
        bit_buffer |= (ch << bit_count)
        bit_count += 11
        while bit_count >= 8:
            out.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bit_count -= 8
    if bit_count > 0:
        out.append(bit_buffer & 0xFF)
    return bytes(out)


def build_rc_channels_frame(channels_11bit) -> bytes:
    payload = pack_rc_channels(channels_11bit)
    frame_len = len(payload) + 2  
    body = bytes([CRSF_TYPE_RC_CHANNELS_PACKED]) + payload
    crc = _crc8(body)
    return bytes([CRSF_SYNC, frame_len]) + body + bytes([crc])


class CRSFSender:
    def __init__(self, port="/dev/ttyAMA0", baudrate=420000):
        self.port = port
        self.baudrate = baudrate
        self._ser = None

    def open(self):
        if serial is None:
            raise RuntimeError(
                "pyserial kurulu degil. 'pip install pyserial --break-system-packages' ile kur."
            )
        if self._ser is None or not self._ser.is_open:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=0)
        return self

    def close(self):
        if self._ser is not None and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def send_channels(self, channels):

        norm = [_normalize_to_crsf(v) for v in channels[:CRSF_NUM_CHANNELS]]
        while len(norm) < CRSF_NUM_CHANNELS:
            norm.append(CRSF_CHANNEL_MID)

        frame = build_rc_channels_frame(norm)

        if self._ser is not None and self._ser.is_open:
            self._ser.write(frame)
        return frame


if __name__ == "__main__":
    test_channels = [0.0, 0.5, -0.5, 1.0]
    sender = CRSFSender()
    frame = sender.send_channels(test_channels)
    print("Uretilen CRSF frame (hex):", frame.hex())
    print("Frame uzunlugu:", len(frame), "byte (beklenen: 26 = sync+len+type+22payload+crc)")
    assert len(frame) == 26, "Frame uzunlugu CRSF spesifikasyonuyla uyusmuyor!"
    assert frame[0] == CRSF_SYNC
    print("Self-test OK - donaniniz baglaninca CRSFSender(port=...).open() ile gercek gonderim baslar.")