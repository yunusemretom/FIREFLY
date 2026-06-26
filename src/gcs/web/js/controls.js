/**
 * FIREFLY GCS — Controls & Settings Module
 */

const Controls = (() => {
  // ── Varsayılan kontrol limitleri ─────────────────────────
  // Ayarlar panelinden değiştirilebilir; server'dan da gelebilir.
  let ctrlLimits = {
    thr_min:  -1.0,
    thr_max:   1.0,
    pitch_min: -1.0,
    pitch_max:  1.0,
    roll_min:  -1.0,
    roll_max:   1.0,
    yaw_min:   -1.0,
    yaw_max:    1.0,
  };

  // Dışarıdan erişim için auto-send callback (App tarafından set edilir)
  let _onControlChange = null;

  /** App.js'ten çağrılır; slider değişince tetiklenecek fonksiyon */
  function setAutoSendCallback(fn) {
    _onControlChange = fn;
  }

  // ── Collapsible panels ──────────────────────────────────
  function initCollapsible(toggleId, bodyId, arrowId, startOpen=false) {
    const toggle = document.getElementById(toggleId);
    const body   = document.getElementById(bodyId);
    const arrow  = document.getElementById(arrowId);
    if(!toggle||!body)return;

    const open  = ()=>{body.classList.remove('hidden');if(arrow)arrow.textContent='▲';toggle.setAttribute('aria-expanded','true');};
    const close = ()=>{body.classList.add('hidden');   if(arrow)arrow.textContent='▼';toggle.setAttribute('aria-expanded','false');};

    if(startOpen) open(); else close();

    toggle.addEventListener('click',()=> body.classList.contains('hidden')?open():close());
    toggle.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();toggle.click();}});
  }

  // ── Sync slider ↔ number input ──────────────────────────
  function linkSliderNum(sliderId, numId) {
    const sl=document.getElementById(sliderId);
    const nm=document.getElementById(numId);
    if(!sl||!nm)return;
    sl.addEventListener('input',()=>{nm.value=parseFloat(sl.value).toFixed(3);});
    nm.addEventListener('input',()=>{sl.value=nm.value;});
  }

  // ── Kontrol slider: göster + otomatik gönder ────────────
  function initControlSlider(id, valId, decimals=2) {
    const sl = document.getElementById(id);
    const vl = document.getElementById(valId);
    if(!sl) return;

    const update = () => {
      const v = parseFloat(sl.value);
      if(vl) vl.textContent = v.toFixed(decimals);
      // Otomatik gönderim
      if(_onControlChange) _onControlChange();
    };

    sl.addEventListener('input', update);
    // İlk değeri göster
    if(vl) vl.textContent = parseFloat(sl.value).toFixed(decimals);
  }

  // ── Slider limitlerini güncelle ─────────────────────────
  function applyLimitsToSlider(sliderId, valId, min, max) {
    const sl = document.getElementById(sliderId);
    if(!sl) return;

    const current = parseFloat(sl.value);
    sl.min = min;
    sl.max = max;

    // Mevcut değer aralık dışına çıktıysa sıkıştır
    const clamped = Math.max(min, Math.min(max, current));
    sl.value = clamped;

    // Gösterim etiketini güncelle
    const vl = document.getElementById(valId);
    if(vl) vl.textContent = clamped.toFixed(2);
  }

  /** Kontrol limitlerini uygula (içeriden ve dışarıdan çağrılabilir) */
  function applyControlLimits(limits) {
    Object.assign(ctrlLimits, limits);

    applyLimitsToSlider('ctrl-thr',   'val-thr',   ctrlLimits.thr_min,   ctrlLimits.thr_max);
    applyLimitsToSlider('ctrl-pitch', 'val-pitch',  ctrlLimits.pitch_min, ctrlLimits.pitch_max);
    applyLimitsToSlider('ctrl-roll',  'val-roll',   ctrlLimits.roll_min,  ctrlLimits.roll_max);
    applyLimitsToSlider('ctrl-yaw',   'val-yaw',    ctrlLimits.yaw_min,   ctrlLimits.yaw_max);
  }

  // ── Ayarlar panelindeki limit input'larını senkronize et ─
  function initLimitInputs() {
    // Min/max giriş alanı ID'leri → hangi limit anahtarı
    const fields = [
      ['s-thr-min',   's-thr-max',   'thr_min',   'thr_max'],
      ['s-pitch-min', 's-pitch-max', 'pitch_min', 'pitch_max'],
      ['s-roll-min',  's-roll-max',  'roll_min',  'roll_max'],
      ['s-yaw-min',   's-yaw-max',   'yaw_min',   'yaw_max'],
    ];

    // Mevcut limit değerlerini input'lara yaz
    const set = (id, v) => { const el = document.getElementById(id); if(el) el.value = v; };
    set('s-thr-min',   ctrlLimits.thr_min);   set('s-thr-max',   ctrlLimits.thr_max);
    set('s-pitch-min', ctrlLimits.pitch_min);  set('s-pitch-max', ctrlLimits.pitch_max);
    set('s-roll-min',  ctrlLimits.roll_min);   set('s-roll-max',  ctrlLimits.roll_max);
    set('s-yaw-min',   ctrlLimits.yaw_min);    set('s-yaw-max',   ctrlLimits.yaw_max);

    // Kaydet butonu
    const btnSave = document.getElementById('btn-save-ctrl-limits');
    if(btnSave) btnSave.addEventListener('click', () => {
      const get = (id, fallback) => {
        const el = document.getElementById(id);
        const v  = el ? parseFloat(el.value) : fallback;
        return isNaN(v) ? fallback : v;
      };

      const newLimits = {
        thr_min:   get('s-thr-min',   -1.0),
        thr_max:   get('s-thr-max',    1.0),
        pitch_min: get('s-pitch-min', -1.0),
        pitch_max: get('s-pitch-max',  1.0),
        roll_min:  get('s-roll-min',  -1.0),
        roll_max:  get('s-roll-max',   1.0),
        yaw_min:   get('s-yaw-min',   -1.0),
        yaw_max:   get('s-yaw-max',   1.0),
      };

      // Basit doğrulama: min < max
      let valid = true;
      for(const [minKey, maxKey] of [['thr_min','thr_max'],['pitch_min','pitch_max'],['roll_min','roll_max'],['yaw_min','yaw_max']]) {
        if(newLimits[minKey] >= newLimits[maxKey]) { valid = false; break; }
      }
      if(!valid) {
        App.showToast && App.showToast('⚠ Min değer Max değerinden büyük olamaz!', 'error');
        return;
      }

      applyControlLimits(newLimits);
      // Server'a da gönder (kalıcılık için)
      if(_onControlChange) App.sendSettings && App.sendSettings({ ctrl_limits: newLimits });
      App.showToast && App.showToast('Kontrol limitleri güncellendi ✓', 'success');
    });

    // Sıfırla butonu
    const btnReset = document.getElementById('btn-reset-ctrl-limits');
    if(btnReset) btnReset.addEventListener('click', () => {
      const defaults = { thr_min:-1, thr_max:1, pitch_min:-1, pitch_max:1, roll_min:-1, roll_max:1, yaw_min:-1, yaw_max:1 };
      set('s-thr-min',   defaults.thr_min);   set('s-thr-max',   defaults.thr_max);
      set('s-pitch-min', defaults.pitch_min);  set('s-pitch-max', defaults.pitch_max);
      set('s-roll-min',  defaults.roll_min);   set('s-roll-max',  defaults.roll_max);
      set('s-yaw-min',   defaults.yaw_min);    set('s-yaw-max',   defaults.yaw_max);
      applyControlLimits(defaults);
      App.showToast && App.showToast('Kontrol limitleri varsayılana döndürüldü', 'info');
    });
  }

  // ── Camera source ───────────────────────────────────────
  let activeCamSrc = null;
  let usbStream = null;

  function initCamera() {
    const radios = document.querySelectorAll('input[name="cam-src"]');
    const ipRow  = document.getElementById('ip-url-row');
    const usbRow = document.getElementById('usb-row');

    radios.forEach(r=>{
      r.addEventListener('change',()=>{
        ipRow.style.display  = r.value==='ip'  ?'flex':'none';
        usbRow.style.display = r.value==='usb' ?'flex':'none';
      });
    });

    const btnScan = document.getElementById('btn-scan-usb');
    if(btnScan) btnScan.addEventListener('click', async ()=>{
      try{
        await navigator.mediaDevices.getUserMedia({video:true});
        const devs = await navigator.mediaDevices.enumerateDevices();
        const sel  = document.getElementById('s-usb-device');
        sel.innerHTML='<option value="">Seçin…</option>';
        devs.filter(d=>d.kind==='videoinput').forEach(d=>{
          const o=document.createElement('option');
          o.value=d.deviceId; o.textContent=d.label||`Kamera ${sel.options.length}`;
          sel.appendChild(o);
        });
      }catch(e){alert('Kamera izni gerekli: '+e.message);}
    });

    const btnApply = document.getElementById('btn-cam-apply');
    if(btnApply) btnApply.addEventListener('click', applyCamera);
  }

  async function applyCamera() {
    const srcEl = document.querySelector('input[name="cam-src"]:checked');
    const src   = srcEl ? srcEl.value : 'none';
    const camImg  = document.getElementById('cam-ip');
    const camVid  = document.getElementById('cam-usb');
    const ph      = document.getElementById('cam-placeholder');
    const srcLbl  = document.getElementById('cam-src-label');

    if(usbStream){usbStream.getTracks().forEach(t=>t.stop());usbStream=null;}
    camImg.style.display='none';
    camVid.style.display='none';
    ph.style.display='flex';
    activeCamSrc=null;

    if(src==='ip'){
      const url=document.getElementById('s-cam-url') && document.getElementById('s-cam-url').value.trim();
      if(!url){alert('MJPEG URL boş olamaz.');return;}
      camImg.src=url;
      camImg.style.display='block';
      ph.style.display='none';
      if(srcLbl)srcLbl.textContent='IP-CAM';
      activeCamSrc='ip';
    } else if(src==='usb'){
      const devId=document.getElementById('s-usb-device') && document.getElementById('s-usb-device').value;
      const constraints={video:devId?{deviceId:{exact:devId}}:true};
      try{
        usbStream=await navigator.mediaDevices.getUserMedia(constraints);
        camVid.srcObject=usbStream;
        camVid.style.display='block';
        ph.style.display='none';
        if(srcLbl)srcLbl.textContent='USB-CAM';
        activeCamSrc='usb';
      }catch(e){alert('USB kamera açılamadı: '+e.message);}
    } else {
      if(srcLbl)srcLbl.textContent='NO SOURCE';
    }
  }

  // ── Kontrol slider değerlerini oku ──────────────────────
  function getControlValues() {
    const get = (id, fallback) => {
      const el = document.getElementById(id);
      return el ? parseFloat(el.value) : fallback;
    };
    return {
      throttle: get('ctrl-thr',    0),
      pitch:    get('ctrl-pitch',  0),
      roll:     get('ctrl-roll',   0),
      yaw:      get('ctrl-yaw',    0),
      max_thr:  get('ctrl-maxthr', 1),
    };
  }

  function getPIDValues() {
    const get = (id, fallback) => {
      const el = document.getElementById(id);
      return el ? parseFloat(el.value) : fallback;
    };
    return {
      kp_yaw:       get('s-kp-yaw',   0.07),
      kp_pitch:     get('s-kp-pitch', 0.035),
      kp_thr:       get('s-kp-thr',   0.01),
      max_throttle: get('s-max-thr',  1.0),
    };
  }

  function getCtrlLimits() { return Object.assign({}, ctrlLimits); }

  function init() {
    // Collapsibles
    initCollapsible('instr-toggle','instr-body','instr-arrow', true);
    initCollapsible('ctrl-toggle', 'ctrl-body',  'ctrl-arrow', false);

    // Settings sliders (PID)
    linkSliderNum('s-kp-yaw-sl',  's-kp-yaw');
    linkSliderNum('s-kp-pitch-sl','s-kp-pitch');
    linkSliderNum('s-kp-thr-sl',  's-kp-thr');
    linkSliderNum('s-max-thr-sl', 's-max-thr');

    // Control sliders — otomatik gönderimli
    initControlSlider('ctrl-thr',    'val-thr');
    initControlSlider('ctrl-pitch',  'val-pitch');
    initControlSlider('ctrl-roll',   'val-roll');
    initControlSlider('ctrl-yaw',    'val-yaw');
    initControlSlider('ctrl-maxthr', 'val-maxthr');

    // Limit inputs
    initLimitInputs();

    // Camera
    initCamera();
  }

  return { init, getControlValues, getPIDValues, getCtrlLimits, setAutoSendCallback, applyControlLimits };
})();
