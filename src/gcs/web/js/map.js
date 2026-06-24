/**
 * FIREFLY GCS — Map View (Canvas 2D)
 * Drone ve hedefi top-down 2D haritada gösterir.
 */

const MapView = (() => {
  let canvas, ctx;
  let dronePos=[0,0,0], targetPos=[0,0,0];
  let droneYaw=0;
  let trail=[], targetTrail=[];
  const MAX_TRAIL=60;

  const C = {
    BG:'#080b0f', GRID:'rgba(0,212,255,0.05)', GRIDBOLD:'rgba(0,212,255,0.12)',
    DRONE:'#00d4ff', TARGET:'#ff6b2b', TRAIL:'rgba(0,212,255,0.3)',
    TTRL:'rgba(255,107,43,0.3)', LINE:'rgba(255,255,255,0.15)',
    TEXT:'#c8d8e8', DIM:'#4a6080', ACC:'#00d4ff'
  };

  let scale=1.0; // pixels per unit
  let offsetX=0, offsetY=0;
  let viewRange=2000; // world units visible

  function worldToScreen(wx,wy){
    return [
      canvas.width/2  + (wx-offsetX)*scale,
      canvas.height/2 - (wy-offsetY)*scale
    ];
  }

  function drawGrid(){
    const step=200; // world units
    ctx.strokeStyle=C.GRID; ctx.lineWidth=1;
    const cols=Math.ceil(canvas.width/scale/step)+2;
    const rows=Math.ceil(canvas.height/scale/step)+2;
    const startX=Math.floor((offsetX-canvas.width/2/scale)/step)*step;
    const startY=Math.floor((offsetY-canvas.height/2/scale)/step)*step;
    for(let i=0;i<=cols;i++){
      const wx=startX+i*step;
      const [sx]=worldToScreen(wx,0);
      const isBold=wx%1000===0;
      ctx.strokeStyle=isBold?C.GRIDBOLD:C.GRID;
      ctx.lineWidth=isBold?1:0.5;
      ctx.beginPath();ctx.moveTo(sx,0);ctx.lineTo(sx,canvas.height);ctx.stroke();
    }
    for(let j=0;j<=rows;j++){
      const wy=startY+j*step;
      const [,sy]=worldToScreen(0,wy);
      const isBold=wy%1000===0;
      ctx.strokeStyle=isBold?C.GRIDBOLD:C.GRID;
      ctx.lineWidth=isBold?1:0.5;
      ctx.beginPath();ctx.moveTo(0,sy);ctx.lineTo(canvas.width,sy);ctx.stroke();
    }
    // Origin cross
    const [ox,oy]=worldToScreen(0,0);
    ctx.strokeStyle='rgba(0,212,255,0.2)';ctx.lineWidth=1;
    ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(ox,0);ctx.lineTo(ox,canvas.height);ctx.stroke();
    ctx.beginPath();ctx.moveTo(0,oy);ctx.lineTo(canvas.width,oy);ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawTrail(pts, color){
    if(pts.length<2)return;
    for(let i=1;i<pts.length;i++){
      const alpha=(i/pts.length)*0.5;
      const [x1,y1]=worldToScreen(pts[i-1][0],pts[i-1][1]);
      const [x2,y2]=worldToScreen(pts[i][0],pts[i][1]);
      ctx.strokeStyle=color.replace('0.3',alpha.toFixed(2));
      ctx.lineWidth=1.5;
      ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
    }
  }

  function drawDrone(sx,sy,yaw){
    ctx.save();ctx.translate(sx,sy);ctx.rotate(yaw*Math.PI/180);
    // Body
    ctx.strokeStyle=C.DRONE;ctx.lineWidth=2;
    ctx.fillStyle='rgba(0,212,255,0.15)';
    ctx.beginPath();ctx.moveTo(0,-12);ctx.lineTo(-8,8);ctx.lineTo(0,4);ctx.lineTo(8,8);ctx.closePath();
    ctx.fill();ctx.stroke();
    // Rotors
    ctx.strokeStyle='rgba(0,212,255,0.4)';ctx.lineWidth=1;
    for(const [rx,ry] of [[-10,-8],[10,-8],[-10,8],[10,8]]){
      ctx.beginPath();ctx.arc(rx,ry,5,0,Math.PI*2);ctx.stroke();
    }
    // Glow
    ctx.shadowColor=C.DRONE;ctx.shadowBlur=12;
    ctx.beginPath();ctx.arc(0,0,4,0,Math.PI*2);ctx.fillStyle=C.DRONE;ctx.fill();
    ctx.restore();
  }

  function drawTarget(sx,sy){
    ctx.save();ctx.translate(sx,sy);
    // Rings
    ctx.strokeStyle=C.TARGET;ctx.lineWidth=2;
    ctx.beginPath();ctx.arc(0,0,14,0,Math.PI*2);ctx.stroke();
    ctx.strokeStyle='rgba(255,107,43,0.4)';ctx.lineWidth=1;
    ctx.beginPath();ctx.arc(0,0,22,0,Math.PI*2);ctx.stroke();
    // Cross
    ctx.strokeStyle=C.TARGET;ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(-8,0);ctx.lineTo(8,0);ctx.stroke();
    ctx.beginPath();ctx.moveTo(0,-8);ctx.lineTo(0,8);ctx.stroke();
    // Glow
    ctx.shadowColor=C.TARGET;ctx.shadowBlur=12;
    ctx.beginPath();ctx.arc(0,0,4,0,Math.PI*2);ctx.fillStyle=C.TARGET;ctx.fill();
    ctx.restore();
  }

  function drawLine(x1,y1,x2,y2){
    ctx.strokeStyle=C.LINE;ctx.lineWidth=1;ctx.setLineDash([6,4]);
    ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawLabel(txt,sx,sy,color='#c8d8e8',yoff=18){
    ctx.fillStyle=color;
    ctx.font='10px Share Tech Mono, monospace';
    ctx.textAlign='center';ctx.textBaseline='top';
    ctx.fillText(txt,sx,sy+yoff);
  }

  function resize(){
    const wrap=canvas.parentElement;
    canvas.width=wrap.clientWidth;
    canvas.height=wrap.clientHeight;
    scale=Math.min(canvas.width,canvas.height)/viewRange;
  }

  function render(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle=C.BG;ctx.fillRect(0,0,canvas.width,canvas.height);

    // Auto-center: keep both drone and target visible
    const cx=(dronePos[0]+targetPos[0])/2;
    const cy=(dronePos[1]+targetPos[1])/2;
    const dx=Math.abs(dronePos[0]-targetPos[0]);
    const dy=Math.abs(dronePos[1]-targetPos[1]);
    const needed=Math.max(dx,dy,400)*1.5;
    viewRange=Math.max(2000,needed);
    scale=Math.min(canvas.width,canvas.height)/viewRange;
    offsetX=cx; offsetY=cy;

    drawGrid();

    const [dsx,dsy]=worldToScreen(dronePos[0],dronePos[1]);
    const [tsx,tsy]=worldToScreen(targetPos[0],targetPos[1]);

    // Line between drone and target
    drawLine(dsx,dsy,tsx,tsy);

    // Trails
    drawTrail(trail,C.TRAIL);
    drawTrail(targetTrail,C.TTRL);

    // Icons
    drawTarget(tsx,tsy);
    drawDrone(dsx,dsy,droneYaw);

    // Labels
    drawLabel('▲ DRONE',dsx,dsy,'#00d4ff');
    drawLabel('◉ HEDEF',tsx,tsy,'#ff6b2b');

    // Scale bar
    const sbLen=100*scale;
    const sbX=24, sbY=canvas.height-24;
    ctx.strokeStyle='rgba(255,255,255,0.3)';ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(sbX,sbY);ctx.lineTo(sbX+sbLen,sbY);ctx.stroke();
    ctx.fillStyle='rgba(255,255,255,0.5)';
    ctx.font='9px Share Tech Mono, monospace';ctx.textAlign='left';ctx.textBaseline='bottom';
    ctx.fillText('100m',sbX,sbY-3);

    // Compass rose top-right
    const crx=canvas.width-40, cry=40;
    ctx.strokeStyle='rgba(0,212,255,0.4)';ctx.lineWidth=1;
    ctx.beginPath();ctx.arc(crx,cry,18,0,Math.PI*2);ctx.stroke();
    ctx.fillStyle='#ff4444';ctx.font='bold 10px monospace';ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.fillText('N',crx,cry-10);
    ctx.fillStyle='rgba(255,255,255,0.4)';ctx.fillText('S',crx,cry+10);
    ctx.fillText('E',crx+12,cry);ctx.fillText('W',crx-12,cry);
  }

  function update(data){
    if(!data) return;
    const dp=data?.drone?.position??[0,0,0];
    const tp=data?.target?.position??[0,0,0];
    const drot=data?.drone?.rotation??[0,0,0];
    dronePos=dp; targetPos=tp; droneYaw=drot[2];

    trail.push([dp[0],dp[1]]);
    if(trail.length>MAX_TRAIL)trail.shift();
    targetTrail.push([tp[0],tp[1]]);
    if(targetTrail.length>MAX_TRAIL)targetTrail.shift();

    // Update coord displays
    const set=(id,v)=>{const el=document.getElementById(id);if(el)el.textContent=typeof v==='number'?v.toFixed(1):v;};
    set('mc-dx',dp[0]);set('mc-dy',dp[1]);set('mc-dz',dp[2]);
    set('mc-ds',data?.drone?.speed??0);
    set('mc-tx',tp[0]);set('mc-ty',tp[1]);set('mc-tz',tp[2]);
    set('mc-ts',data?.target?.speed??0);

    // Distance
    const dist3d=Math.sqrt((dp[0]-tp[0])**2+(dp[1]-tp[1])**2+(dp[2]-tp[2])**2);
    const bearing=Math.atan2(tp[0]-dp[0],tp[1]-dp[1])*180/Math.PI;
    const dl=document.getElementById('map-dist-label');
    const bl=document.getElementById('map-bearing-label');
    if(dl)dl.textContent=`Mesafe: ${(dist3d/100).toFixed(1)} m`;
    if(bl)bl.textContent=`Yön: ${((bearing+360)%360).toFixed(0)}°`;

    if(canvas)render();
  }

  function init(){
    canvas=document.getElementById('map-canvas');
    if(!canvas)return;
    ctx=canvas.getContext('2d');
    resize();
    window.addEventListener('resize',resize);
    render();
  }

  return {init,update,resize};
})();
