/**
 * FIREFLY GCS — Main App (WebSocket + Navigation + Telemetry)
 */

const App = (() => {
  const WS_URL             = 'ws://localhost:8765';
  const RECONNECT_MS       = 3000;  // Yeniden bağlanma gecikmesi (ms)
  const CONNECT_TIMEOUT_MS = 5000;  // Bağlantı timeout süresi (ms)

  let ws = null;
  let armed = false;
  let autoMode = false;
  let lastTelemetry = null;
  let reconnectTimer = null;
  let battery = null; // not in SDK, shown as N/A

  // ── Page Navigation ──────────────────────────────────────
  function initNav() {
    document.querySelectorAll('.nav-tab').forEach(btn=>{
      btn.addEventListener('click',()=>{
        const page=btn.dataset.page;
        document.querySelectorAll('.nav-tab').forEach(b=>{b.classList.remove('active');b.setAttribute('aria-selected','false');});
        document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
        btn.classList.add('active');btn.setAttribute('aria-selected','true');
        const pg=document.getElementById('page-'+page);
        if(pg){pg.classList.add('active');}
        // Re-render map if switching to maps
        if(page==='maps'){MapView.resize();if(lastTelemetry)MapView.update(lastTelemetry);}
      });
    });
  }

  // ── WebSocket ────────────────────────────────────────────
  /** Sunucu (WS) bağlantı durumu — Ayarlar sekmesinde gösterilir */
  function setServerStatus(state) { // 'disconnected'|'connecting'|'connected'
    const dot   = document.getElementById('ss-dot');
    const stEl  = document.getElementById('ss-state');
    const urlEl = document.getElementById('ss-url');
    const bar   = document.getElementById('server-status-bar');
    if(!dot) return;
    const cls = { connecting: 'connecting', connected: 'connected' };
    dot.className = 'ss-dot' + (cls[state] ? ' ' + cls[state] : '');
    if(stEl)  stEl.textContent  = { connecting: 'Bağlanıyor…', connected: 'Bağlı', disconnected: 'Bağlantı yok' }[state];
    if(urlEl) urlEl.textContent = state === 'connected' ? WS_URL : '';
    if(bar)   bar.dataset.state = state;
  }

  /** Drone (TCP) telemetri durumu — Topbar'da gösterilir */
  function setDroneStatus(connected) {
    const dot = document.getElementById('drone-dot');
    const lbl = document.getElementById('drone-label');
    if(!dot || !lbl) return;
    if(connected) {
      dot.className = 'ws-dot connected';
      lbl.textContent = 'DRONE BAĞLI';
    } else {
      dot.className = 'ws-dot';
      lbl.textContent = 'DRONE YOK';
    }
  }

  function connect() {
    if(ws&&(ws.readyState===WebSocket.OPEN||ws.readyState===WebSocket.CONNECTING))return;
    clearTimeout(reconnectTimer);
    setServerStatus('connecting');

    let connectTimeoutId = null;
    let didOpen = false;

    ws=new WebSocket(WS_URL);

    // ── Bağlantı Timeout ──────────────────────────────────
    // WebSocket API yerleşik timeout desteği sunmaz.
    // CONNECT_TIMEOUT_MS içinde OPEN olmadıysa bağlantıyı iptal et.
    connectTimeoutId = setTimeout(() => {
      if (!didOpen && ws && ws.readyState !== WebSocket.OPEN) {
        ws.close(); // onclose tetiklenir → yeniden deneme
        setServerStatus('disconnected');
        showToast(`⚠ Sunucuya bağlanılamadı (${WS_URL}) — Sunucu çalışıyor mu?`, 'error');
        console.warn('[GCS] WS bağlantı timeout:', WS_URL);
      }
    }, CONNECT_TIMEOUT_MS);

    ws.onopen=()=>{
      didOpen = true;
      clearTimeout(connectTimeoutId);
      setServerStatus('connected');
      console.log('[GCS] WS bağlandı');
    };

    ws.onclose=()=>{
      clearTimeout(connectTimeoutId);
      setServerStatus('disconnected');
      ws=null;
      reconnectTimer=setTimeout(connect, RECONNECT_MS);
    };

    ws.onerror=(e)=>{
      clearTimeout(connectTimeoutId);
      console.warn('[GCS] WS bağlantı hatası', e);
      // onclose her zaman onerror'dan sonra tetiklenir — toast burada göster
      if (!didOpen) {
        showToast(`⚠ Sunucuya bağlanılamadı (${WS_URL})`, 'error');
      }
    };

    ws.onmessage=(e)=>{
      try{
        const msg=JSON.parse(e.data);
        if(msg.type==='telemetry') handleTelemetry(msg.data);
        else if(msg.type==='settings') applySettings(msg.data);
        else if(msg.type==='connect_result') handleConnectResult(msg);
        else if(msg.type==='settings_saved') showToast('Ayarlar kaydedildi ✓');
      }catch(err){console.error('[GCS] mesaj parse hatası',err);}
    };
  }

  function send(obj){
    if(ws&&ws.readyState===WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  // ── Drone Connect / Disconnect ───────────────────────────
  function initConnectBtn() {
    const btn=document.getElementById('btn-drone-connect');
    let droneConnected=false;
    if(!btn)return;
    btn.addEventListener('click',()=>{
      if(!droneConnected){
        const host=(document.getElementById('s-host')||{value:'127.0.0.1'}).value;
        const port=parseInt((document.getElementById('s-port')||{value:12345}).value);
        send({type:'connect_drone',host,port});
        btn.textContent='BAĞLANIYOR…';btn.disabled=true;
      } else {
        send({type:'disconnect_drone'});
        droneConnected=false;
        btn.textContent='BAĞLAN';
        updateConnStatus(false);
        setDroneStatus(false);
      }
    });
    window._setDroneConnected=(v)=>{
      droneConnected=v;
      btn.textContent=v?'KES':'BAĞLAN';
      btn.disabled=false;
      updateConnStatus(v);
    };
  }

  function handleConnectResult(msg){
    const ok=msg.success;
    window._setDroneConnected && window._setDroneConnected(ok);
    setDroneStatus(ok);
    showToast(msg.message||(ok?'Bağlandı':'Bağlantı başarısız'), ok?'success':'error');
  }

  function updateConnStatus(connected){
    const demoTag=document.getElementById('demo-tag');
    if(demoTag) demoTag.classList.toggle('hidden',connected);
  }

  // ── ARM / DISARM ─────────────────────────────────────────
  function initArmBtn() {
    const btn=document.getElementById('btn-arm-main');
    if(!btn)return;
    btn.addEventListener('click',()=>{
      armed=!armed;
      btn.dataset.armed=armed;
      btn.textContent=armed?'ARMED':'DISARM';
      send({type:'set_arm',value:armed});
      // Update mode chip color
      (function(){var e=document.getElementById('chip-mode');if(e)e.classList.toggle('armed',armed);})();
    });
  }

  // ── Kamikaze toggle ──────────────────────────────────────
  function initKamikazeBtn() {
    const btn=document.getElementById('btn-kamikaze');
    if(!btn)return;
    btn.addEventListener('click',()=>{
      autoMode=!autoMode;
      btn.dataset.mode=autoMode?'kamikaze':'manual';
      document.getElementById('mode-text').textContent=autoMode?'🎯 KAMİKAZE':'🎯 MANUEL';
      document.getElementById('lbl-mode').textContent=autoMode?'KAMİKAZE':'MANUEL';
      const chip=document.getElementById('chip-mode');
      if(chip) if(chip)chip.classList.toggle('kamikaze',autoMode);
      document.getElementById('hud-mode').textContent=autoMode?'KAMIKAZE':'MANUAL';
      send({type:'set_auto_mode',value:autoMode});
    });
  }

  // ── Send control inputs ──────────────────────────────────
  function sendControlNow() {
    const v = Controls.getControlValues();
    send({ type: 'control', throttle: v.throttle, pitch: v.pitch, roll: v.roll, yaw: v.yaw, arm: armed });
  }

  function initSendBtn() {
    const btn = document.getElementById('btn-send-ctrl');
    if(btn) btn.addEventListener('click', sendControlNow);
  }

  // ── Save settings ────────────────────────────────────────
  function sendSettings(data) {
    send({ type: 'update_settings', data: data });
  }

  function initSaveSettings() {
    // PID + connection together
    const btnPid = document.getElementById('btn-save-settings');
    if (btnPid) btnPid.addEventListener('click', () => {
      const data = {
        ...Controls.getPIDValues(),
      };
      send({ type: 'update_settings', data });
    });

    // Connection (host, port, timeout)
    const btnConn = document.getElementById('btn-save-conn');
    if (btnConn) btnConn.addEventListener('click', () => {
      const host    = (function(){var e=document.getElementById('s-host');return e?e.value.trim():'127.0.0.1';})();
      const port    = parseInt((document.getElementById('s-port')||{value:12345}).value);
      const timeout = parseFloat((document.getElementById('s-timeout')||{value:2}).value);
      send({ type: 'update_settings', data: { sim_host: host, sim_port: port, connect_timeout: timeout } });
      showToast('Bağlantı ayarları kaydedildi — ' + host + ':' + port + ' (timeout: ' + timeout + 's)', 'success');
    });

    // Map base GPS
    const btnMap = document.getElementById('btn-save-map');
    if (btnMap) btnMap.addEventListener('click', () => {
      const lat = parseFloat((document.getElementById('s-base-lat')||{value:39.9255}).value);
      const lon = parseFloat((document.getElementById('s-base-lon')||{value:32.8660}).value);
      if (isNaN(lat) || isNaN(lon)) { showToast('Geçersiz GPS koordinatı', 'error'); return; }
      MapView.setBaseGPS(lat, lon);
      showToast(`Harita merkezi: ${lat.toFixed(4)}, ${lon.toFixed(4)}`, 'success');
    });
  }

  // ── Apply server settings to UI ──────────────────────────
  function applySettings(s){
    const setVal=(id,v)=>{const el=document.getElementById(id);if(el)el.value=v;};
    if(s.kp_yaw!==undefined){setVal('s-kp-yaw',s.kp_yaw);setVal('s-kp-yaw-sl',s.kp_yaw);}
    if(s.kp_pitch!==undefined){setVal('s-kp-pitch',s.kp_pitch);setVal('s-kp-pitch-sl',s.kp_pitch);}
    if(s.kp_thr!==undefined){setVal('s-kp-thr',s.kp_thr);setVal('s-kp-thr-sl',s.kp_thr);}
    if(s.max_throttle!==undefined){setVal('s-max-thr',s.max_throttle);setVal('s-max-thr-sl',s.max_throttle);}
    if(s.camera_url){setVal('s-cam-url',s.camera_url);}
    if(s.sim_host){setVal('s-host',s.sim_host);}
    if(s.sim_port){setVal('s-port',s.sim_port);}
    if(s.connect_timeout){setVal('s-timeout',s.connect_timeout);}
  }

  // ── Telemetry update ─────────────────────────────────────
  function handleTelemetry(data){
    lastTelemetry=data;
    setDroneStatus(data && data.connected === true);

    const d=(data&&data.drone)||{};
    const t=(data&&data.target)||{};
    const pos=d.position||[0,0,0];
    const rot=d.rotation||[0,0,0];
    const spd=d.speed||0;
    const alt=d.altitude||0;
    const tpos=t.position||[0,0,0];
    const tspd=t.speed||0;

    const fmt=(v,n=1)=>typeof v==='number'?v.toFixed(n):'---';
    const setT=(id,v)=>{const el=document.getElementById(id);if(el)el.textContent=v;};

    setT('t-pos',`${fmt(pos[0])}, ${fmt(pos[1])}, ${fmt(pos[2])}`);
    setT('t-rot',`${fmt(rot[0])}, ${fmt(rot[1])}, ${fmt(rot[2])}`);
    setT('t-spd',`${fmt(spd,0)} cm/s`);
    setT('t-alt',`${fmt(alt,0)} cm`);
    setT('t-tpos',`${fmt(tpos[0])}, ${fmt(tpos[1])}, ${fmt(tpos[2])}`);
    setT('t-tspd',`${fmt(tspd,0)} cm/s`);

    const dist=Math.sqrt((pos[0]-tpos[0])**2+(pos[1]-tpos[1])**2+(pos[2]-tpos[2])**2);
    setT('t-dist',`${(dist/100).toFixed(1)} m`);

    // HUD pos overlay
    setT('hud-pos',`POS ${fmt(pos[0],0)}, ${fmt(pos[1],0)}, ${fmt(pos[2],0)}`);

    // Battery (not in SDK, placeholder)
    setT('lbl-battery', battery!=null?`${battery}%`:'N/A');

    // Mission phase
    let phase='BEKLEME', lockTxt='YOK', apprTxt='---';
    if(autoMode && data && data.connected!==false){
      if(dist>4500){phase='FAZ 1 — YAKALAMA';lockTxt='AKTİF';apprTxt='AGRESIF';}
      else if(dist>0){phase='FAZ 2 — YAKLAŞIM';lockTxt='KİLİTLİ';apprTxt='HASSAS';}
    }
    setT('m-phase', phase);
    setT('m-lock', lockTxt);
    setT('m-appr', apprTxt);

    // Lock indicator
    const lock=document.getElementById('lock-indicator');
    if(lock) lock.classList.toggle('hidden', lockTxt==='YOK');

    // Demo tag — artık idle durumu gösterir
    const demoTag=document.getElementById('demo-tag');
    if(demoTag){
      const isIdle = data && data.idle === true;
      const isConnected = data && data.connected === true;
      demoTag.textContent = isConnected ? '' : 'BAĞLI DEĞİL';
      demoTag.classList.toggle('hidden', isConnected);
    }

    // Instruments
    Instruments.update(data);

    // Map (if on maps page)
    const mapsActive=(function(){var e=document.getElementById('page-maps');return e&&e.classList.contains('active');})();
    if(mapsActive) MapView.update(data);
  }

  // ── Toast notification ───────────────────────────────────
  function showToast(msg, type='info'){
    let t=document.getElementById('gcs-toast');
    if(!t){
      t=document.createElement('div');t.id='gcs-toast';
      Object.assign(t.style,{
        position:'fixed',bottom:'20px',right:'20px',
        padding:'10px 18px',borderRadius:'6px',
        fontFamily:'Share Tech Mono, monospace',fontSize:'12px',
        transition:'opacity 0.3s',zIndex:'9999',
        border:'1px solid',
      });
      document.body.appendChild(t);
    }
    const colors={success:['#00ff88','#002a1a'],error:['#ff2b2b','#2a0000'],info:['#00d4ff','#001a2a']};
    const [fg,bg]=colors[type]||colors.info;
    t.style.color=fg;t.style.background=bg;t.style.borderColor=fg;
    t.textContent=msg;t.style.opacity='1';
    clearTimeout(t._timer);
    t._timer=setTimeout(()=>{t.style.opacity='0';},3000);
  }

  // ── Init ──────────────────────────────────────────────────
  function init(){
    initNav();
    Controls.init();
    Instruments.init();
    MapView.init();
    initConnectBtn();
    initArmBtn();
    initKamikazeBtn();
    initSendBtn();
    initSaveSettings();
    connect();

    // Kontrol slider'ları değişince otomatik gönder
    Controls.setAutoSendCallback(sendControlNow);

    // Resize map on window resize
    window.addEventListener('resize', function() {
      const mp = document.getElementById('page-maps');
      if(mp && mp.classList.contains('active')) MapView.resize();
    });

    console.log('[GCS] FIREFLY Ground Control Station başlatıldı');
  }

  return { init: init, showToast: showToast, sendSettings: sendSettings };
})();

document.addEventListener('DOMContentLoaded', App.init);
