/**
 * FIREFLY GCS — World Map (Leaflet.js + OpenStreetMap)
 *
 * Sim koordinatları (cm, Unreal Engine) → GPS offset dönüşümü:
 *   baseLat/baseLon ayarlardan gelir (varsayılan: Ankara)
 *   X (cm) → Boylam offset, Y (cm) → Enlem offset
 *   1 derece enlem ≈ 111,320 m = 11,132,000 cm
 *   1 derece boylam ≈ 111,320 * cos(lat) m
 */

const MapView = (() => {
  let map = null;
  let droneMarker = null;
  let targetMarker = null;
  let droneTrail = null;
  let targetTrail = null;
  let distanceLine = null;

  let baseLat = 39.9255;  // Ankara Esenboğa civarı (varsayılan)
  let baseLon = 32.8660;

  const MAX_TRAIL = 80;
  let dronePoints = [];
  let targetPoints = [];

  // ── Custom SVG icons ──────────────────────────────────────
  const droneIcon = L.divIcon({
    className: '',
    html: `<div class="lf-drone-icon" id="lf-drone-icon">
      <svg width="32" height="32" viewBox="0 0 32 32">
        <polygon points="16,2 8,28 16,22 24,28" fill="#00d4ff" stroke="#00d4ff" stroke-width="1" opacity="0.9"/>
        <circle cx="16" cy="16" r="3" fill="#fff"/>
      </svg>
    </div>`,
    iconSize: [32, 32],
    iconAnchor: [16, 16],
  });

  const targetIcon = L.divIcon({
    className: '',
    html: `<div class="lf-target-icon">
      <svg width="32" height="32" viewBox="0 0 32 32">
        <circle cx="16" cy="16" r="12" fill="none" stroke="#ff6b2b" stroke-width="2"/>
        <circle cx="16" cy="16" r="5" fill="none" stroke="#ff6b2b" stroke-width="2"/>
        <line x1="4" y1="16" x2="28" y2="16" stroke="#ff6b2b" stroke-width="1.5"/>
        <line x1="16" y1="4" x2="16" y2="28" stroke="#ff6b2b" stroke-width="1.5"/>
        <circle cx="16" cy="16" r="2" fill="#ff6b2b"/>
      </svg>
    </div>`,
    iconSize: [32, 32],
    iconAnchor: [16, 16],
  });

  // ── Coordinate conversion ─────────────────────────────────
  // sim X → longitude offset, sim Y → latitude offset
  // Sim birim = cm
  function simToLatLon(x, y) {
    const CM_PER_DEG_LAT = 111320 * 100; // ~11,132,000 cm/derece
    const CM_PER_DEG_LON = Math.cos(baseLat * Math.PI / 180) * CM_PER_DEG_LAT;
    const lat = baseLat + (y / CM_PER_DEG_LAT);
    const lon = baseLon + (x / CM_PER_DEG_LON);
    return [lat, lon];
  }

  // ── Dark tile layer ───────────────────────────────────────
  function getDarkTiles() {
    // CartoDB Dark Matter — ücretsiz, kayıt gerektirmez
    return L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OSM</a> © <a href="https://carto.com/">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 20,
      opacity: 0.85,
    });
  }

  // ── Init ──────────────────────────────────────────────────
  function init() {
    const el = document.getElementById('leaflet-map');
    if (!el || map) return;

    map = L.map('leaflet-map', {
      center: [baseLat, baseLon],
      zoom: 15,
      zoomControl: true,
      attributionControl: true,
    });

    getDarkTiles().addTo(map);

    // Drone marker
    droneMarker = L.marker([baseLat, baseLon], { icon: droneIcon, zIndexOffset: 1000 })
      .addTo(map)
      .bindTooltip('DRONE', { permanent: false, direction: 'top', className: 'lf-tooltip' });

    // Target marker
    targetMarker = L.marker([baseLat, baseLon], { icon: targetIcon, zIndexOffset: 900 })
      .addTo(map)
      .bindTooltip('HEDEF', { permanent: false, direction: 'top', className: 'lf-tooltip' });

    // Trails (polylines)
    droneTrail = L.polyline([], {
      color: '#00d4ff', weight: 2, opacity: 0.5, dashArray: null,
    }).addTo(map);

    targetTrail = L.polyline([], {
      color: '#ff6b2b', weight: 2, opacity: 0.4, dashArray: '4,4',
    }).addTo(map);

    // Distance line
    distanceLine = L.polyline([], {
      color: 'rgba(255,255,255,0.2)', weight: 1, dashArray: '6,4',
    }).addTo(map);

    // Crosshair at base point
    L.circle([baseLat, baseLon], {
      radius: 50, color: 'rgba(0,212,255,0.2)', fillColor: 'transparent', weight: 1,
    }).addTo(map);

    // Base point marker
    L.circleMarker([baseLat, baseLon], {
      radius: 5, color: '#00d4ff', fillColor: '#00d4ff', fillOpacity: 0.3, weight: 2,
    }).addTo(map).bindTooltip('ORIGIN', { className: 'lf-tooltip', direction: 'right' });

    injectLeafletStyles();
  }

  // ── Update with telemetry data ─────────────────────────────
  function update(data) {
    if (!map) return;

    const dp = data?.drone?.position ?? [0, 0, 0];
    const tp = data?.target?.position ?? [0, 0, 0];
    const drot = data?.drone?.rotation ?? [0, 0, 0];

    const dLatLon = simToLatLon(dp[0], dp[1]);
    const tLatLon = simToLatLon(tp[0], tp[1]);

    // Move markers
    droneMarker.setLatLng(dLatLon);
    targetMarker.setLatLng(tLatLon);

    // Rotate drone icon based on yaw
    const iconEl = document.getElementById('lf-drone-icon');
    if (iconEl) iconEl.style.transform = `rotate(${drot[2]}deg)`;

    // Update trails
    dronePoints.push(dLatLon);
    if (dronePoints.length > MAX_TRAIL) dronePoints.shift();
    droneTrail.setLatLngs(dronePoints);

    targetPoints.push(tLatLon);
    if (targetPoints.length > MAX_TRAIL) targetPoints.shift();
    targetTrail.setLatLngs(targetPoints);

    // Distance line
    distanceLine.setLatLngs([dLatLon, tLatLon]);

    // Update coord cards
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = typeof v === 'number' ? v.toFixed(1) : v; };
    set('mc-dx', dp[0]); set('mc-dy', dp[1]); set('mc-dz', dp[2]);
    set('mc-ds', data?.drone?.speed ?? 0);
    set('mc-tx', tp[0]); set('mc-ty', tp[1]); set('mc-tz', tp[2]);
    set('mc-ts', data?.target?.speed ?? 0);

    // Distance & bearing labels
    const dist3d = Math.sqrt((dp[0] - tp[0]) ** 2 + (dp[1] - tp[1]) ** 2 + (dp[2] - tp[2]) ** 2);
    const bearing = (Math.atan2(tp[0] - dp[0], tp[1] - dp[1]) * 180 / Math.PI + 360) % 360;
    const dl = document.getElementById('map-dist-label');
    const bl = document.getElementById('map-bearing-label');
    if (dl) dl.textContent = `Mesafe: ${(dist3d / 100).toFixed(1)} m`;
    if (bl) bl.textContent = `Yön: ${bearing.toFixed(0)}°`;
  }

  // ── Re-center map ─────────────────────────────────────────
  function resize() {
    if (map) {
      setTimeout(() => map.invalidateSize(), 100);
    }
  }

  // ── Update base GPS from settings ─────────────────────────
  function setBaseGPS(lat, lon) {
    baseLat = lat;
    baseLon = lon;
    dronePoints = [];
    targetPoints = [];
    if (map) {
      map.setView([lat, lon], map.getZoom());
    }
  }

  // ── Inject Leaflet custom styles ──────────────────────────
  function injectLeafletStyles() {
    const style = document.createElement('style');
    style.textContent = `
      /* Leaflet dark theme tweaks */
      .leaflet-container { background: #080b0f; font-family: 'Share Tech Mono', monospace; }
      .leaflet-control-zoom a {
        background: #111820 !important; color: #00d4ff !important;
        border-color: #1e2d3d !important; font-weight: bold;
      }
      .leaflet-control-zoom a:hover { background: #1a2332 !important; }
      .leaflet-control-attribution {
        background: rgba(8,11,15,0.7) !important; color: #4a6080 !important; font-size: 9px;
      }
      .leaflet-control-attribution a { color: #4a6080; }
      .lf-tooltip {
        background: rgba(8,11,15,0.9) !important;
        border: 1px solid #1e2d3d !important;
        color: #00d4ff !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 10px !important;
        padding: 2px 8px !important;
        border-radius: 3px !important;
        box-shadow: none !important;
      }
      .lf-drone-icon svg { filter: drop-shadow(0 0 6px #00d4ff); }
      .lf-target-icon svg { filter: drop-shadow(0 0 6px #ff6b2b); animation: targetPulse 1.5s ease-in-out infinite; }
      @keyframes targetPulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
    `;
    document.head.appendChild(style);
  }

  return { init, update, resize, setBaseGPS };
})();
