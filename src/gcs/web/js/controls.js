/**
 * FIREFLY GCS — Controls & Settings Module
 */

const Controls = (() => {
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

  // ── Slider display values ───────────────────────────────
  function initSlider(id, valId, decimals=2) {
    const sl=document.getElementById(id);
    const vl=document.getElementById(valId);
    if(!sl||!vl)return;
    const update=()=>{vl.textContent=parseFloat(sl.value).toFixed(decimals);};
    sl.addEventListener('input',update);
    update();
  }

  // ── Camera source ───────────────────────────────────────
  let activeCamSrc = null; // 'ip' | 'usb' | null
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

    // Scan USB cameras
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

    // Apply camera
    const btnApply = document.getElementById('btn-cam-apply');
    if(btnApply) btnApply.addEventListener('click', applyCamera);
  }

  async function applyCamera() {
    const src = document.querySelector('input[name="cam-src"]:checked')?.value??'none';
    const camImg  = document.getElementById('cam-ip');
    const camVid  = document.getElementById('cam-usb');
    const ph      = document.getElementById('cam-placeholder');
    const srcLbl  = document.getElementById('cam-src-label');

    // Stop previous USB stream
    if(usbStream){usbStream.getTracks().forEach(t=>t.stop());usbStream=null;}
    camImg.style.display='none';
    camVid.style.display='none';
    ph.style.display='flex';
    activeCamSrc=null;

    if(src==='ip'){
      const url=document.getElementById('s-cam-url')?.value?.trim();
      if(!url){alert('MJPEG URL boş olamaz.');return;}
      camImg.src=url;
      camImg.style.display='block';
      ph.style.display='none';
      if(srcLbl)srcLbl.textContent='IP-CAM';
      activeCamSrc='ip';
    } else if(src==='usb'){
      const devId=document.getElementById('s-usb-device')?.value;
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

  // ── Controls panel send ─────────────────────────────────
  function getControlValues() {
    return {
      throttle: parseFloat(document.getElementById('ctrl-thr')?.value??0),
      pitch:    parseFloat(document.getElementById('ctrl-pitch')?.value??0),
      roll:     parseFloat(document.getElementById('ctrl-roll')?.value??0),
      yaw:      parseFloat(document.getElementById('ctrl-yaw')?.value??0),
      max_thr:  parseFloat(document.getElementById('ctrl-maxthr')?.value??1),
    };
  }

  function getPIDValues() {
    return {
      kp_yaw:      parseFloat(document.getElementById('s-kp-yaw')?.value??0.07),
      kp_pitch:    parseFloat(document.getElementById('s-kp-pitch')?.value??0.035),
      kp_thr:      parseFloat(document.getElementById('s-kp-thr')?.value??0.01),
      max_throttle:parseFloat(document.getElementById('s-max-thr')?.value??1.0),
    };
  }

  function init() {
    // Collapsibles
    initCollapsible('instr-toggle','instr-body','instr-arrow', true);
    initCollapsible('ctrl-toggle', 'ctrl-body',  'ctrl-arrow', false);

    // Settings sliders
    linkSliderNum('s-kp-yaw-sl',  's-kp-yaw');
    linkSliderNum('s-kp-pitch-sl','s-kp-pitch');
    linkSliderNum('s-kp-thr-sl',  's-kp-thr');
    linkSliderNum('s-max-thr-sl', 's-max-thr');

    // Control sliders display
    initSlider('ctrl-thr',    'val-thr');
    initSlider('ctrl-pitch',  'val-pitch');
    initSlider('ctrl-roll',   'val-roll');
    initSlider('ctrl-yaw',    'val-yaw');
    initSlider('ctrl-maxthr', 'val-maxthr');

    // Camera
    initCamera();
  }

  return { init, getControlValues, getPIDValues };
})();
