/**
 * FIREFLY GCS — Main App (WebSocket + Navigation + Telemetry)
 */

const App = (() => {
  const WS_URL = 'ws://localhost:8765';
  const RECONNECT_MS = 3000;

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
  function setWsStatus(state) { // 'disconnected'|'connecting'|'connected'
    const dot=document.getElementById('ws-dot');
    const lbl=document.getElementById('ws-label');
    if(!dot||!lbl)return;
    dot.className='ws-dot'+({connecting:' connecting',connected:' connected'}[state]??'');
    lbl.textContent={connecting:'BAĞLANIYOR…',connected:'BAĞLI',disconnected:'BAĞLANTI YOK'}[state];
  }

  function connect() {
    if(ws&&(ws.readyState===WebSocket.OPEN||ws.readyState===WebSocket.CONNECTING))return;
    clearTimeout(reconnectTimer);
    setWsStatus('connecting');
    ws=new WebSocket(WS_URL);

    ws.onopen=()=>{
      setWsStatus('connected');
      console.log('[GCS] WS bağlandı');
    };

    ws.onclose=()=>{
      setWsStatus('disconnected');
      ws=null;
      reconnectTimer=setTimeout(connect, RECONNECT_MS);
    };

    ws.onerror=(e)=>{console.warn('[GCS] WS hata',e);};

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
        const host=document.getElementById('s-host')?.value??'127.0.0.1';
        const port=parseInt(document.getElementById('s-port')?.value??12345);
        send({type:'connect_drone',host,port});
        btn.textContent='BAĞLANIYOR…';btn.disabled=true;
      } else {
        send({type:'disconnect_drone'});
        droneConnected=false;
        btn.textContent='BAĞLAN';
        updateConnStatus(false);
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
    window._setDroneConnected?.(ok);
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
      document.getElementById('chip-mode')?.classList.toggle('armed',armed);
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
      if(chip) chip.classList.toggle('kamikaze',autoMode);
      document.getElementById('hud-mode').textContent=autoMode?'KAMIKAZE':'MANUAL';
      send({type:'set_auto_mode',value:autoMode});
    });
  }

  // ── Send control inputs ──────────────────────────────────
  function initSendBtn() {
    const btn=document.getElementById('btn-send-ctrl');
    if(!btn)return;
    btn.addEventListener('click',()=>{
      const v=Controls.getControlValues();
      send({type:'control', throttle:v.throttle, pitch:v.pitch, roll:v.roll, yaw:v.yaw, arm:armed});
    });
  }

  // ── Save settings ────────────────────────────────────────
  function initSaveSettings() {
    const btn=document.getElementById('btn-save-settings');
    if(!btn)return;
    btn.addEventListener('click',()=>{
      const data=Controls.getPIDValues();
      send({type:'update_settings',data});
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
  }

  // ── Telemetry update ─────────────────────────────────────
  function handleTelemetry(data){
    lastTelemetry=data;

    const d=data?.drone??{};
    const t=data?.target??{};
    const pos=d.position??[0,0,0];
    const rot=d.rotation??[0,0,0];
    const spd=d.speed??0;
    const alt=d.altitude??0;
    const tpos=t.position??[0,0,0];
    const tspd=t.speed??0;

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
    if(autoMode && data?.connected!==false){
      if(dist>4500){phase='FAZ 1 — YAKALAMA';lockTxt='AKTİF';apprTxt='AGRESIF';}
      else if(dist>0){phase='FAZ 2 — YAKLAŞIM';lockTxt='KİLİTLİ';apprTxt='HASSAS';}
    }
    setT('m-phase', phase);
    setT('m-lock', lockTxt);
    setT('m-appr', apprTxt);

    // Lock indicator
    const lock=document.getElementById('lock-indicator');
    if(lock) lock.classList.toggle('hidden', lockTxt==='YOK');

    // Demo tag
    const demoTag=document.getElementById('demo-tag');
    if(demoTag) demoTag.classList.toggle('hidden', data?.demo===false&&data?.connected===true);

    // Instruments
    Instruments.update(data);

    // Map (if on maps page)
    const mapsActive=document.getElementById('page-maps')?.classList.contains('active');
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
    const [fg,bg]=colors[type]??colors.info;
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

    // Resize map on window resize
    window.addEventListener('resize',()=>{
      if(document.getElementById('page-maps')?.classList.contains('active')) MapView.resize();
    });

    console.log('[GCS] FIREFLY Ground Control Station başlatıldı');
  }

  return { init };
})();

document.addEventListener('DOMContentLoaded', App.init);
