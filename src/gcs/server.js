/**
 * ================================================================================
 * FIREFLY GCS — Node.js WebSocket + HTTP + TCP Drone Köprüsü
 * ================================================================================
 * Unreal Engine simülatörüne doğrudan TCP bağlantısı kurar.
 * drone_sdk.py protokolünü Node.js'te uygular.
 *
 * TCP Protokolü:
 *   GELEN  (Simülatör→Biz): "f0,f1,...,f17\n"  (18 float, CSV)
 *     v[0-2]  : Drone Position (X, Y, Z)
 *     v[3-5]  : Drone Rotation (Roll, Pitch, Yaw)
 *     v[6-8]  : Drone Velocity (X, Y, Z)
 *     v[9]    : Drone Speed (cm/s)
 *     v[10]   : Altitude
 *     v[11-13]: Target Position (X, Y, Z)
 *     v[14-16]: Target Rotation (Roll, Pitch, Yaw)
 *     v[17]   : Target Speed (cm/s)
 *
 *   GİDEN  (Biz→Simülatör): "throttle,pitch,roll,yaw,arm\n"
 *     arm: 1 veya 0
 *
 * Kullanım:
 *     node src/gcs/server.js
 *   veya
 *     cd src/gcs && npm start
 *
 * Port yapılandırması (env değişkenleri veya varsayılan):
 *     GCS_HTTP_PORT = 8080   → Web arayüzü
 *     GCS_WS_PORT   = 8765   → WebSocket (tarayıcı ↔ bu sunucu)
 * ================================================================================
 */

'use strict';

const express = require('express');
const http    = require('http');
const net     = require('net');
const { WebSocketServer, WebSocket } = require('ws');
const path    = require('path');

// ─── Port yapılandırması ───────────────────────────────────────────────────────
const HTTP_PORT = parseInt(process.env.GCS_HTTP_PORT || '8080', 10);
const WS_PORT   = parseInt(process.env.GCS_WS_PORT   || '8765', 10);

const TELEMETRY_HZ       = 20;    // Saniyede tarayıcıya telemetri gönderim sayısı
const CONNECT_TIMEOUT_MS = 5000;  // TCP bağlantı timeout (ms)
const INACTIVITY_MS      = 5000;  // Son veriden bu kadar süre geçerse bağlantıyı kes (ms)

const WEB_DIR = path.join(__dirname, 'web');

// ─── Loglama ──────────────────────────────────────────────────────────────────
const log = {
  info:  (...a) => console.log('[GCS INFO]', ...a),
  warn:  (...a) => console.warn('[GCS WARN]', ...a),
  error: (...a) => console.error('[GCS ERROR]', ...a),
};

// ─── Global durum ─────────────────────────────────────────────────────────────
/** @type {Set<WebSocket>} */
const wsClients = new Set();

let currentSettings = {
  kp_yaw:          0.07,
  kp_pitch:         0.035,
  kp_thr:           0.01,
  max_throttle:     1.0,
  camera_url:       '',
  sim_host:         '127.0.0.1',
  sim_port:         12345,
  connect_timeout:  5.0,
};

// ─── Drone TCP bağlantı durumu ────────────────────────────────────────────────
const drone = {
  socket:          null,   // net.Socket
  connected:       false,
  buffer:          '',     // gelen veri tamponu
  inactivityTimer: null,   // son veri alındıktan sonraki timer

  // Son bilinen kontrol değerleri
  controls: {
    throttle: 0.0,
    pitch:    0.0,
    roll:     0.0,
    yaw:      0.0,
    arm:      false,
  },

  // Son telemetri verisi
  telemetry: {
    connected: false,
    idle:      true,
    drone: {
      position: [0.0, 0.0, 0.0],
      rotation: [0.0, 0.0, 0.0],
      velocity: [0.0, 0.0, 0.0],
      speed:    0.0,
      altitude: 0.0,
    },
    target: {
      position: [0.0, 0.0, 0.0],
      rotation: [0.0, 0.0, 0.0],
      speed:    0.0,
    },
    controls: {
      throttle:  0.0,
      pitch:     0.0,
      roll:      0.0,
      yaw:       0.0,
      arm:       false,
      auto_mode: false,
    },
  },
};

// ─── Telemetri parse — drone_sdk.py _parse_telemetry() ile aynı mantık ────────
function parseTelemetryLine(line) {
  try {
    const parts = line.trim().split(',');
    if (parts.length < 18) return;

    const v = parts.map(Number);
    if (v.some(isNaN)) return;

    drone.telemetry.connected = true;
    drone.telemetry.idle      = false;

    drone.telemetry.drone.position = [v[0],  v[1],  v[2]];
    drone.telemetry.drone.rotation = [v[3],  v[4],  v[5]];
    drone.telemetry.drone.velocity = [v[6],  v[7],  v[8]];
    drone.telemetry.drone.speed    = v[9];
    drone.telemetry.drone.altitude = v[10];

    drone.telemetry.target.position = [v[11], v[12], v[13]];
    drone.telemetry.target.rotation = [v[14], v[15], v[16]];
    drone.telemetry.target.speed    = v[17];

    drone.telemetry.controls = { ...drone.controls };
  } catch (err) {
    log.warn('Telemetri parse hatası:', err.message);
  }
}

// ─── Kontrol komutu gönder — drone_sdk.py send_inputs() ile aynı format ───────
function sendDroneInputs(throttle, pitch, roll, yaw, arm) {
  if (!drone.connected || !drone.socket) return;

  // Değerleri sınırla
  const thr  = Math.max(-1.0, Math.min(1.0,  throttle));  // [-1=alçal, 0=hover, 1=tırman]
  const pit  = Math.max(-1.0, Math.min(1.0,  pitch));
  const rol  = Math.max(-1.0, Math.min(1.0,  roll));
  const yw   = Math.max(-1.0, Math.min(1.0,  yaw));
  const armV = arm ? 1 : 0;

  // Kontrol değerlerini kaydet
  drone.controls = { throttle: thr, pitch: pit, roll: rol, yaw: yw, arm: Boolean(arm) };

  const msg = `${thr.toFixed(4)},${pit.toFixed(4)},${rol.toFixed(4)},${yw.toFixed(4)},${armV}\n`;
  try {
    drone.socket.write(msg);
  } catch (err) {
    log.warn('Kontrol komutu gönderilemedi:', err.message);
    drone.connected = false;
  }
}

// ─── Inactivity timer (veri gelmezse bağlantıyı kes) ─────────────────────────
function resetInactivityTimer() {
  clearTimeout(drone.inactivityTimer);
  drone.inactivityTimer = setTimeout(() => {
    if (drone.connected) {
      log.warn(`Simülatörden ${INACTIVITY_MS / 1000}s boyunca veri gelmedi — bağlantı kesiliyor.`);
      closeDroneConnection('timeout');
    }
  }, INACTIVITY_MS);
}

// ─── Drone TCP bağlantısını kapat ─────────────────────────────────────────────
function closeDroneConnection(reason = 'manual') {
  clearTimeout(drone.inactivityTimer);

  if (drone.socket) {
    try { drone.socket.destroy(); } catch { /* ignore */ }
    drone.socket = null;
  }
  drone.connected = false;
  drone.buffer    = '';

  // Telemetriyi idle'a sıfırla
  drone.telemetry.connected = false;
  drone.telemetry.idle      = true;

  // Tüm WS istemcilerine bildir
  broadcastToClients(JSON.stringify({
    type:    'connect_result',
    success: false,
    message: reason === 'timeout'
      ? `⚠ Bağlantı kesildi — ${INACTIVITY_MS / 1000}s boyunca veri gelmedi`
      : 'Bağlantı kesildi',
  }));

  log.info(`Drone TCP bağlantısı kapatıldı (sebep: ${reason})`);
}

// ─── Drone TCP bağlantısı kur ─────────────────────────────────────────────────
function connectToDrone(host, port) {
  if (drone.connected) {
    closeDroneConnection('reconnect');
  }

  log.info(`Simülatöre bağlanılıyor: ${host}:${port} (timeout: ${CONNECT_TIMEOUT_MS}ms)`);

  const sock = new net.Socket();
  drone.socket = sock;
  drone.buffer = '';

  // Bağlantı timeout
  const connectTimer = setTimeout(() => {
    if (!drone.connected) {
      log.warn(`TCP bağlantı timeout: ${host}:${port}`);
      sock.destroy();
      drone.socket    = null;
      drone.connected = false;
      broadcastToClients(JSON.stringify({
        type:    'connect_result',
        success: false,
        message: `⚠ Bağlantı zaman aşımı — ${host}:${port} yanıt vermedi`,
      }));
    }
  }, CONNECT_TIMEOUT_MS);

  sock.connect(port, host, () => {
    clearTimeout(connectTimer);
    drone.connected = true;
    log.info(`Simülatöre bağlandı: ${host}:${port}`);

    // Ayarları güncelle
    currentSettings.sim_host = host;
    currentSettings.sim_port = port;

    // WS istemcilerine bildir
    broadcastToClients(JSON.stringify({
      type:    'connect_result',
      success: true,
      message: `Bağlandı: ${host}:${port}`,
    }));

    // İnactivity timer'ı başlat
    resetInactivityTimer();
  });

  // Gelen telemetri verisi
  sock.on('data', (chunk) => {
    resetInactivityTimer();
    drone.buffer += chunk.toString('utf8');

    // Satır satır işle
    let newlineIdx;
    while ((newlineIdx = drone.buffer.indexOf('\n')) !== -1) {
      const line = drone.buffer.slice(0, newlineIdx);
      drone.buffer = drone.buffer.slice(newlineIdx + 1);
      if (line.trim()) {
        parseTelemetryLine(line);
      }
    }
  });

  sock.on('error', (err) => {
    clearTimeout(connectTimer);
    log.warn(`TCP bağlantı hatası: ${err.message}`);
    drone.socket    = null;
    drone.connected = false;
    clearTimeout(drone.inactivityTimer);
    drone.telemetry.connected = false;
    drone.telemetry.idle      = true;

    broadcastToClients(JSON.stringify({
      type:    'connect_result',
      success: false,
      message: `⚠ Bağlantı hatası: ${err.message}`,
    }));
  });

  sock.on('close', () => {
    clearTimeout(connectTimer);
    if (drone.connected) {
      log.info('TCP bağlantısı kapandı.');
      closeDroneConnection('closed');
    }
  });
}

// ─── Tüm WS istemcilerine mesaj yayınla ───────────────────────────────────────
function broadcastToClients(payload) {
  for (const client of wsClients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(payload, (err) => {
        if (err) wsClients.delete(client);
      });
    } else {
      wsClients.delete(client);
    }
  }
}

// ─── Telemetri yayıncısı (20Hz) ───────────────────────────────────────────────
function startTelemetryBroadcast() {
  const intervalMs = Math.round(1000 / TELEMETRY_HZ);
  setInterval(() => {
    if (wsClients.size === 0) return;
    broadcastToClients(JSON.stringify({ type: 'telemetry', data: drone.telemetry }));
  }, intervalMs);
}

// ─── WS istemci mesaj işleyicisi ─────────────────────────────────────────────
function handleWsMessage(ws, raw) {
  let msg;
  try {
    msg = JSON.parse(raw);
  } catch {
    log.warn('Geçersiz JSON:', String(raw).slice(0, 80));
    return;
  }

  const mtype = msg.type || '';

  switch (mtype) {

    case 'connect_drone': {
      const host = String(msg.host || currentSettings.sim_host).trim();
      const port = parseInt(msg.port  || currentSettings.sim_port, 10);
      connectToDrone(host, port);
      break;
    }

    case 'disconnect_drone': {
      closeDroneConnection('manual');
      break;
    }

    case 'control': {
      if (drone.connected) {
        sendDroneInputs(
          parseFloat(msg.throttle || 0),
          parseFloat(msg.pitch    || 0),
          parseFloat(msg.roll     || 0),
          parseFloat(msg.yaw      || 0),
          Boolean(msg.arm),
        );
      }
      break;
    }

    case 'set_arm': {
      if (drone.connected) {
        const arm = Boolean(msg.value);
        sendDroneInputs(
          drone.controls.throttle,
          drone.controls.pitch,
          drone.controls.roll,
          drone.controls.yaw,
          arm,
        );
      }
      break;
    }

    case 'set_auto_mode': {
      drone.telemetry.controls.auto_mode = Boolean(msg.value);
      log.info(`Otonom mod: ${drone.telemetry.controls.auto_mode ? 'AKTİF' : 'PASİF'}`);
      break;
    }

    case 'update_settings': {
      const incoming = msg.data || {};
      if ('sim_port' in incoming)       incoming.sim_port       = parseInt(incoming.sim_port, 10);
      if ('connect_timeout' in incoming) incoming.connect_timeout = parseFloat(incoming.connect_timeout);
      Object.assign(currentSettings, incoming);
      log.info('Ayarlar güncellendi:', JSON.stringify(incoming));
      ws.send(JSON.stringify({ type: 'settings_saved', success: true }));
      break;
    }

    case 'ping': {
      ws.send(JSON.stringify({ type: 'pong' }));
      break;
    }

    default:
      log.warn('Bilinmeyen mesaj tipi:', mtype);
  }
}

// ─── WebSocket sunucusu ───────────────────────────────────────────────────────
function createWebSocketServer() {
  const wss = new WebSocketServer({ port: WS_PORT, host: 'localhost' });

  wss.on('listening', () => {
    log.info(`WebSocket dinleniyor: ws://localhost:${WS_PORT}`);
  });

  wss.on('error', (err) => {
    log.error('WebSocket sunucu hatası:', err.message);
    if (err.code === 'EADDRINUSE') {
      log.error(`Port ${WS_PORT} kullanımda! GCS_WS_PORT env ile değiştirin.`);
      process.exit(1);
    }
  });

  wss.on('connection', (ws, req) => {
    const addr = `${req.socket.remoteAddress}:${req.socket.remotePort}`;
    wsClients.add(ws);
    log.info(`Tarayıcı bağlandı: ${addr}  (toplam: ${wsClients.size})`);

    // Bağlantı anında mevcut ayarları ve drone durumunu gönder
    ws.send(JSON.stringify({ type: 'settings', data: currentSettings }));

    ws.on('message', (raw) => {
      try { handleWsMessage(ws, raw); }
      catch (err) { log.error('Mesaj işleme hatası:', err.message); }
    });

    ws.on('close', () => {
      wsClients.delete(ws);
      log.info(`Tarayıcı ayrıldı: ${addr}  (toplam: ${wsClients.size})`);
    });

    ws.on('error', (err) => {
      log.warn(`WS istemci hatası (${addr}):`, err.message);
      wsClients.delete(ws);
    });
  });
}

// ─── HTTP / Express sunucusu ──────────────────────────────────────────────────
function createHttpServer() {
  const app = express();

  app.use((req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Cache-Control', 'no-cache');
    next();
  });

  app.use(express.static(WEB_DIR));

  app.get('/', (req, res) => {
    res.sendFile(path.join(WEB_DIR, 'index.html'));
  });

  app.use((req, res) => {
    res.status(404).send('404 — Sayfa bulunamadı');
  });

  const server = http.createServer(app);

  server.on('error', (err) => {
    log.error('HTTP sunucu hatası:', err.message);
    if (err.code === 'EADDRINUSE') {
      log.error(`Port ${HTTP_PORT} kullanımda! GCS_HTTP_PORT env ile değiştirin.`);
      process.exit(1);
    }
  });

  server.listen(HTTP_PORT, () => {
    log.info(`HTTP sunucusu: http://localhost:${HTTP_PORT}`);
  });

  return server;
}

// ─── Başlatma ─────────────────────────────────────────────────────────────────
function main() {
  log.info('='.repeat(60));
  log.info('  🔥 FIREFLY GCS — Node.js  (HTTP + WS + TCP Drone Köprüsü)');
  log.info('='.repeat(60));
  log.info(`  Web Arayüzü: http://localhost:${HTTP_PORT}`);
  log.info(`  WebSocket:   ws://localhost:${WS_PORT}`);
  log.info(`  Drone TCP:   ${currentSettings.sim_host}:${currentSettings.sim_port}  (ayarlar panelinden değiştir)`);
  log.info(`  Timeout:     TCP bağlantı ${CONNECT_TIMEOUT_MS / 1000}s | İnaktivite ${INACTIVITY_MS / 1000}s`);
  log.info('='.repeat(60));

  createHttpServer();
  createWebSocketServer();
  startTelemetryBroadcast();
}

// ─── Graceful shutdown ────────────────────────────────────────────────────────
process.on('SIGINT', () => {
  log.info('GCS sunucusu kapatılıyor… (SIGINT)');
  if (drone.connected) closeDroneConnection('shutdown');
  for (const c of wsClients) { try { c.close(); } catch { /* ignore */ } }
  process.exit(0);
});

process.on('SIGTERM', () => {
  log.info('GCS sunucusu kapatılıyor… (SIGTERM)');
  process.exit(0);
});

main();
