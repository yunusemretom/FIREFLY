/**
 * FIREFLY GCS — Flight Instruments (HTML5 Canvas)
 * Attitude Indicator, Altimeter, Airspeed, Heading/Compass
 */

const Instruments = (() => {
  const C = { BG:'#080b0f', RIM:'#1e2d3d', TEXT:'#c8d8e8', DIM:'#4a6080', ACC:'#00d4ff', WARN:'#ffcc00', DANGER:'#ff2b2b', SKY:'#0a2a4a', GND:'#3a1a0a' };

  function circle(ctx, x, y, r, fill, stroke, sw=1) {
    ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2);
    if(fill){ctx.fillStyle=fill;ctx.fill();}
    if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=sw;ctx.stroke();}
  }
  function line(ctx,x1,y1,x2,y2,color,w=1){
    ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);
    ctx.strokeStyle=color;ctx.lineWidth=w;ctx.stroke();
  }
  function text(ctx,str,x,y,color,size=10,font='Share Tech Mono, monospace',align='center'){
    ctx.fillStyle=color;ctx.font=`${size}px ${font}`;ctx.textAlign=align;ctx.textBaseline='middle';ctx.fillText(str,x,y);
  }

  // ── Attitude Indicator ──────────────────────────────────
  function drawAttitude(canvas, roll=0, pitch=0) {
    const ctx = canvas.getContext('2d');
    const w=canvas.width, h=canvas.height, cx=w/2, cy=h/2, r=w*0.44;
    ctx.clearRect(0,0,w,h);

    // Clipping circle
    ctx.save();
    ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.clip();

    // Horizon
    ctx.save();ctx.translate(cx,cy);ctx.rotate(roll*Math.PI/180);
    const py = pitch*1.8;
    ctx.fillStyle=C.SKY;ctx.fillRect(-w,-h,w*2,h+py);
    ctx.fillStyle=C.GND;ctx.fillRect(-w,py,w*2,h);
    // Horizon line
    ctx.strokeStyle='rgba(255,255,255,0.5)';ctx.lineWidth=1.5;
    ctx.beginPath();ctx.moveTo(-w,py);ctx.lineTo(w,py);ctx.stroke();
    // Pitch lines
    for(let p=-30;p<=30;p+=10){
      if(p===0)continue;
      const ly=py+p*1.8;
      const lw=p%20===0?24:16;
      ctx.strokeStyle='rgba(255,255,255,0.4)';ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(-lw/2,ly);ctx.lineTo(lw/2,ly);ctx.stroke();
      if(p%20===0)text(ctx,p.toString(),-lw/2-14,ly,'rgba(255,255,255,0.5)',8);
    }
    ctx.restore();

    // Roll arc
    ctx.save();ctx.translate(cx,cy);
    ctx.strokeStyle='rgba(0,212,255,0.3)';ctx.lineWidth=1;
    ctx.beginPath();ctx.arc(0,0,r-6,-Math.PI*0.75,Math.PI*-0.25);ctx.stroke();
    for(const deg of[-45,-30,-20,-10,0,10,20,30,45]){
      const a=(deg-90)*Math.PI/180;const ir=r-12,or=r-6;
      ctx.strokeStyle='rgba(0,212,255,0.5)';ctx.lineWidth=deg%30===0?2:1;
      ctx.beginPath();
      ctx.moveTo(ir*Math.cos(a),ir*Math.sin(a));
      ctx.lineTo(or*Math.cos(a),or*Math.sin(a));
      ctx.stroke();
    }
    // Roll pointer
    ctx.rotate(roll*Math.PI/180);
    ctx.fillStyle=C.ACC;
    ctx.beginPath();ctx.moveTo(0,-(r-8));ctx.lineTo(-5,-(r-18));ctx.lineTo(5,-(r-18));ctx.closePath();ctx.fill();
    ctx.restore();

    ctx.restore();// end clip

    // Fixed aircraft symbol
    ctx.strokeStyle='#fff';ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(cx-24,cy);ctx.lineTo(cx-10,cy);ctx.stroke();
    ctx.beginPath();ctx.moveTo(cx+10,cy);ctx.lineTo(cx+24,cy);ctx.stroke();
    circle(ctx,cx,cy,4,'#fff',null);

    // Rim
    circle(ctx,cx,cy,r,null,C.RIM,2);
    circle(ctx,cx,cy,r+3,null,'rgba(0,212,255,0.1)',1);

    // Values
    text(ctx,`R:${roll.toFixed(1)}°`,cx,h-10,C.ACC,9);
  }

  // ── Altimeter ──────────────────────────────────────────
  function drawAltitude(canvas, altCm=0) {
    const ctx=canvas.getContext('2d');
    const w=canvas.width,h=canvas.height,cx=w/2,cy=h/2,r=w*0.44;
    ctx.clearRect(0,0,w,h);

    circle(ctx,cx,cy,r,'#050a0f',C.RIM,2);

    const altM=altCm/100;
    const maxAlt=500;

    // Tick marks
    for(let i=0;i<40;i++){
      const a=(i/40)*Math.PI*2-Math.PI/2;
      const isMajor=i%4===0;
      const ir=r*(isMajor?0.70:0.76);
      const or=r*0.88;
      ctx.strokeStyle=isMajor?C.TEXT:C.DIM;ctx.lineWidth=isMajor?2:1;
      ctx.beginPath();
      ctx.moveTo(cx+ir*Math.cos(a),cy+ir*Math.sin(a));
      ctx.lineTo(cx+or*Math.cos(a),cy+or*Math.sin(a));
      ctx.stroke();
      if(isMajor){
        const val=Math.round((i/40)*maxAlt);
        const tr=r*0.58;
        text(ctx,val.toString(),cx+tr*Math.cos(a),cy+tr*Math.sin(a),C.DIM,8);
      }
    }

    // Hand
    const a=(altM/maxAlt)*Math.PI*2-Math.PI/2;
    ctx.save();ctx.translate(cx,cy);ctx.rotate(a+Math.PI/2);
    ctx.strokeStyle=C.ACC;ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(0,6);ctx.lineTo(0,-r*0.65);ctx.stroke();
    ctx.restore();

    circle(ctx,cx,cy,6,C.ACC,null);
    text(ctx,'ALT',cx,cy-16,C.DIM,8);
    text(ctx,`${altM.toFixed(1)}m`,cx,cy+16,C.ACC,10);
    circle(ctx,cx,cy,r,null,'rgba(0,212,255,0.08)',1);
  }

  // ── Airspeed ───────────────────────────────────────────
  function drawSpeed(canvas, speedCms=0) {
    const ctx=canvas.getContext('2d');
    const w=canvas.width,h=canvas.height,cx=w/2,cy=h/2,r=w*0.44;
    ctx.clearRect(0,0,w,h);

    circle(ctx,cx,cy,r,'#050a0f',C.RIM,2);

    const maxSpd=2000;
    // Color arcs
    const arcSt=-Math.PI*0.75, arcEnd=Math.PI*0.75;
    const greenEnd=arcSt+(arcEnd-arcSt)*0.6;
    const yellowEnd=arcSt+(arcEnd-arcSt)*0.85;
    ctx.lineWidth=5;
    function arc(ctx,from,to,color){
      ctx.beginPath();ctx.arc(cx,cy,r*0.82,from,to);
      ctx.strokeStyle=color;ctx.stroke();
    }
    arc(ctx,arcSt,greenEnd,'rgba(0,255,136,0.25)');
    arc(ctx,greenEnd,yellowEnd,'rgba(255,204,0,0.25)');
    arc(ctx,yellowEnd,arcEnd,'rgba(255,43,43,0.25)');

    // Ticks
    for(let i=0;i<=20;i++){
      const t=i/20;
      const a=arcSt+t*(arcEnd-arcSt);
      const isMajor=i%4===0;
      const ir=r*(isMajor?0.68:0.76);
      ctx.strokeStyle=isMajor?C.TEXT:C.DIM;ctx.lineWidth=isMajor?2:1;
      ctx.beginPath();
      ctx.moveTo(cx+ir*Math.cos(a),cy+ir*Math.sin(a));
      ctx.lineTo(cx+r*0.88*Math.cos(a),cy+r*0.88*Math.sin(a));
      ctx.stroke();
      if(isMajor){
        const val=Math.round(t*maxSpd/100)*100;
        const tr=r*0.55;
        text(ctx,val.toString(),cx+tr*Math.cos(a),cy+tr*Math.sin(a),C.DIM,8);
      }
    }

    // Hand
    const t=Math.min(speedCms/maxSpd,1);
    const angle=arcSt+t*(arcEnd-arcSt);
    ctx.save();ctx.translate(cx,cy);ctx.rotate(angle+Math.PI/2);
    ctx.strokeStyle=C.ACC;ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(0,8);ctx.lineTo(0,-r*0.62);ctx.stroke();
    ctx.restore();

    circle(ctx,cx,cy,6,C.ACC,null);
    text(ctx,'SPD',cx,cy-14,C.DIM,8);
    text(ctx,`${(speedCms/100).toFixed(1)}m/s`,cx,cy+14,C.ACC,9);
  }

  // ── Compass ────────────────────────────────────────────
  function drawCompass(canvas, yaw=0) {
    const ctx=canvas.getContext('2d');
    const w=canvas.width,h=canvas.height,cx=w/2,cy=h/2,r=w*0.44;
    ctx.clearRect(0,0,w,h);

    circle(ctx,cx,cy,r,'#050a0f',C.RIM,2);

    ctx.save();ctx.translate(cx,cy);ctx.rotate(-yaw*Math.PI/180);
    // Cardinals
    const dirs=['N','NE','E','SE','S','SW','W','NW'];
    for(let i=0;i<36;i++){
      const a=i/36*Math.PI*2-Math.PI/2;
      const isMajor=i%9===0;const isCard=i%9===0;
      const ir=r*(isMajor?0.68:0.78);
      ctx.strokeStyle=i===0?C.DANGER:isMajor?C.TEXT:C.DIM;
      ctx.lineWidth=isMajor?2:1;
      ctx.beginPath();
      ctx.moveTo(ir*Math.cos(a),ir*Math.sin(a));
      ctx.lineTo(r*0.88*Math.cos(a),r*0.88*Math.sin(a));
      ctx.stroke();
      if(isCard){
        const lbl=dirs[i/9];
        const tr=r*0.56;
        ctx.fillStyle=i===0?C.DANGER:C.TEXT;
        ctx.font=`bold 10px Share Tech Mono, monospace`;
        ctx.textAlign='center';ctx.textBaseline='middle';
        ctx.fillText(lbl,tr*Math.cos(a),tr*Math.sin(a));
      }
    }
    ctx.restore();

    // Fixed heading pointer
    ctx.strokeStyle=C.ACC;ctx.lineWidth=3;
    ctx.beginPath();ctx.moveTo(cx,cy-r*0.88);ctx.lineTo(cx-5,cy-r*0.7);ctx.lineTo(cx+5,cy-r*0.7);ctx.closePath();
    ctx.fillStyle=C.ACC;ctx.fill();

    circle(ctx,cx,cy,6,C.ACC,null);
    text(ctx,`${Math.round(yaw)}°`,cx,cy+16,C.ACC,10);
  }

  // ── Public API ─────────────────────────────────────────
  function update(data) {
    const rot = data?.drone?.rotation ?? [0,0,0];
    const [roll,pitch,yaw] = rot;
    const spd = data?.drone?.speed ?? 0;
    const alt = data?.drone?.altitude ?? 0;

    drawAttitude(document.getElementById('c-att'), roll, pitch);
    drawAltitude(document.getElementById('c-alt'), alt);
    drawSpeed(document.getElementById('c-spd'), spd);
    drawCompass(document.getElementById('c-hdg'), yaw);
  }

  function init() {
    // Draw with zeros on startup
    update(null);
  }

  return { init, update };
})();
