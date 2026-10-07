#!/usr/bin/env python3
"""
Dựng lại mô hình CAD của tay mới (newarm) từ các file STL thành MỘT trang HTML
xem 3D (three.js), có thanh kéo góc khớp, để đối chiếu với tay thật.

Dùng đúng chuỗi khớp + phép đặt lưới của newarm_make_sim.py (đã sửa 2 lỗi
export), nên hình trong trang trùng với mô hình Gazebo và fk_newarm.py.

    python3 newarm_make_viewer.py            # -> docs/newarm_cad_viewer.html
    python3 newarm_make_viewer.py --body-only out.html   # không bọc <!doctype> (để publish)
"""
import argparse
import base64
import json
import math
from pathlib import Path

import numpy as np

import fk_newarm as F
import newarm_make_sim as S

# link -> (loại, mô tả bộ phận thật)
PARTS = {
    "base_link": ("bracket", "Tấm đế — bắt vào khung, đứng yên"),
    "DigitalServo8120_1": ("servo", "Servo J1 (base) — TD-8120MG, kênh PCA9685 số 0. Thân gắn cứng vào đế"),
    "rds3120_1": ("servo", "Servo J2 (shoulder) — RDS3120, kênh 1. Cả thân servo quay theo J1"),
    "rds3120_support_1": ("bracket", "Giá đỡ đầu ra servo RDS3120 — quay theo J2"),
    "link1_1": ("link", "Khâu 1 (bắp tay)"),
    "Servo_mg996_BracketM_1": ("bracket", "Giá bắt servo MG996R ở khuỷu"),
    "MG996R_servo_1": ("servo", "Servo J3 (elbow) — MG996R, kênh 2"),
    "banhrang_mg996_1": ("gear", "Bánh răng/sừng đầu ra của servo J3"),
    "ServoBracketU_1": ("bracket", "Càng chữ U ở khuỷu — quay theo J3"),
    "link2_1": ("link", "Khâu 2 (cẳng tay)"),
    "MG996R_servo__1__1": ("servo", "Servo J4 (wrist_roll) — MG996R, kênh 3"),
    "banhrang_mg996__1__1": ("gear", "Bánh răng/sừng đầu ra của servo J4"),
    "hopbut_1": ("penbox", "Hộp bút — gắn cứng vào đầu ra J4, nơi dán marker"),
    "but_1": ("pen", "Bút — đồng trục với J4"),
}
JOINT_INFO = {
    "base": ("J1", "TD-8120MG", 0),
    "shoulder": ("J2", "RDS3120", 1),
    "elbow": ("J3", "MG996R", 2),
    "wrist_roll": ("J4", "MG996R", 3),
}

PAGE = r"""<title>Mô hình CAD tay newarm</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --bg:#f4f5f6; --panel:#ffffff; --ink:#14181d; --ink2:#56616c; --line:#dde1e5;
  --accent:#d9571e; --sel:#1f6fd0; --stage:#e9ecef; --chip:#eef1f4;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#12151a; --panel:#1a1f26; --ink:#e8ecf0; --ink2:#9aa5b1; --line:#2c343e;
  --accent:#ff8a4c; --sel:#6aa9f5; --stage:#0d1014; --chip:#232a33; color-scheme:dark } }
:root[data-theme="dark"]{
  --bg:#12151a; --panel:#1a1f26; --ink:#e8ecf0; --ink2:#9aa5b1; --line:#2c343e;
  --accent:#ff8a4c; --sel:#6aa9f5; --stage:#0d1014; --chip:#232a33; color-scheme:dark }
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:14px/1.45 "IBM Plex Sans",system-ui,sans-serif;margin:0;padding-inline:16px;padding-block:16px 28px}
.wrap{max-width:1320px;margin:0 auto;display:flex;flex-direction:column;gap:14px}
h1{font-size:20px;font-weight:600;margin:0;text-wrap:balance}
.sub{color:var(--ink2);margin:2px 0 0;max-width:75ch}
.main{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:14px;align-items:start}
@media (max-width:900px){.main{grid-template-columns:minmax(0,1fr)}}
.stage{position:relative;background:var(--stage);border:1px solid var(--line);border-radius:8px;overflow:hidden;min-width:0}
#gl{display:block;width:100%;height:min(74vh,720px);min-height:380px;touch-action:none;cursor:grab}
.views{position:absolute;left:10px;top:10px;display:flex;flex-wrap:wrap;gap:6px}
.lbl{position:absolute;transform:translate(-50%,-50%);pointer-events:none;white-space:nowrap;
  font:500 11px "IBM Plex Mono",ui-monospace,monospace;padding:1px 5px;border-radius:3px;
  background:var(--panel);color:var(--ink);border:1px solid var(--line)}
.lbl.j{color:var(--accent);border-color:var(--accent)}
#tip{position:absolute;left:10px;bottom:10px;right:10px;background:var(--panel);border:1px solid var(--line);
  border-radius:6px;padding:8px 10px;max-width:520px}
#tip b{font-family:"IBM Plex Mono",ui-monospace,monospace;font-weight:500}
#tip span{color:var(--ink2);display:block}
.side{display:flex;flex-direction:column;gap:14px;min-width:0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;min-width:0}
h2{font-size:12px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--ink2);margin:0 0 10px}
.jrow{display:grid;grid-template-columns:1fr auto;gap:2px 8px;margin-bottom:10px}
.jrow label{font-weight:500}
.jrow small{color:var(--ink2);font-weight:400}
.jrow output{font:500 13px "IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;text-align:right}
.jrow input{grid-column:1/-1;width:100%;accent-color:var(--accent)}
.jrow .servo{grid-column:1/-1;color:var(--ink2);font:12px "IBM Plex Mono",ui-monospace,monospace}
.btns{display:flex;flex-wrap:wrap;gap:6px}
button,select{font:inherit;color:var(--ink);background:var(--chip);border:1px solid var(--line);border-radius:5px;padding:5px 9px;cursor:pointer}
button:hover{border-color:var(--ink2)}
.tog{display:flex;flex-direction:column;gap:5px}
.tog label{display:flex;gap:8px;align-items:center}
.parts{display:flex;flex-direction:column;gap:1px;max-height:300px;overflow:auto}
.parts label{display:flex;gap:8px;align-items:center;padding:3px 4px;border-radius:4px;cursor:pointer;font:12px "IBM Plex Mono",ui-monospace,monospace}
.parts label:hover,.parts label.on{background:var(--chip)}
.parts label.on{outline:1px solid var(--sel)}
.sw{width:10px;height:10px;border-radius:2px;flex:none;border:1px solid var(--line)}
.read{font:12.5px "IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;color:var(--ink2);margin:8px 0 0}
.read b{color:var(--ink);font-weight:500}
ol{margin:0;padding-left:18px;max-width:70ch} li{margin-bottom:5px}
#err{color:var(--accent)}
</style>

<div class="wrap">
  <header>
    <h1>Mô hình CAD tay newarm — dựng lại từ lưới STL</h1>
    <p class="sub">14 chi tiết của bản thiết kế, đặt đúng vị trí và nối bằng 4 khớp. Kéo chuột để xoay, lăn để phóng to, chuột phải để dời. Rê chuột lên chi tiết để biết nó là gì trên tay thật.</p>
  </header>
  <div class="main">
    <div class="stage" id="stage">
      <canvas id="gl"></canvas>
      <div class="views">
        <button data-view="iso">Chéo</button><button data-view="front">Trước</button>
        <button data-view="side">Bên</button><button data-view="top">Trên</button>
      </div>
      <div id="tip"><b id="tipName">—</b><span id="tipDesc">Rê chuột lên một chi tiết, hoặc bấm để ghim.</span></div>
    </div>
    <div class="side">
      <div class="card">
        <h2>Góc khớp</h2>
        <div id="joints"></div>
        <div class="btns">
          <button id="bHome">Home (tay thẳng xuống)</button>
          <button id="bCad">Tư thế trong file CAD</button>
          <button id="bDraw">Tư thế vẽ</button>
        </div>
        <p class="read">Lắp khuỷu:
          <select id="elbowWin"><option value="30">lệch sừng, [-30°,150°]</option><option value="90">sừng giữa, ±90°</option></select>
        </p>
        <p class="read" id="tipRead"></p>
      </div>
      <div class="card">
        <h2>Hiển thị</h2>
        <div class="tog">
          <label><input type="checkbox" id="tAxes" checked> Trục khớp và chiều quay dương</label>
          <label><input type="checkbox" id="tDims" checked> Kích thước giữa các trục khớp (mm)</label>
          <label><input type="checkbox" id="tGhost"> Làm mờ tất cả trừ chi tiết đang chọn</label>
        </div>
      </div>
      <div class="card">
        <h2>Chi tiết (bỏ tích để ẩn)</h2>
        <div class="parts" id="parts"></div>
      </div>
    </div>
  </div>
  <div class="card">
    <h2>Cách đối chiếu với tay thật</h2>
    <ol>
      <li>Bấm <b>Home</b>: đây là tư thế khi cả 4 servo nhận lệnh ứng với q = 0 — tay treo thẳng xuống, bút chĩa xuống. Tay thật ở lệnh home phải giống hệt hình.</li>
      <li>Mũi tên cam ở mỗi khớp là <b>chiều quay dương</b> của mô hình. Tăng lệnh servo vài độ: nếu tay thật quay ngược mũi tên thì khớp đó phải đặt <code>inverted</code> trong bước bring-up.</li>
      <li>Ba số kích thước (J2→J3, J3→J4, J4→đầu bút) đo trong mặt phẳng tay. Đo bằng thước trên tay thật giữa tâm các trục servo để kiểm.</li>
      <li>Các file STL gốc được xuất ở tư thế tay duỗi ngang (nút <b>Tư thế trong file CAD</b>), còn chuỗi khớp trong URDF lại là tay treo thẳng — trang này đã ghép hai thứ lại cho khớp nhau.</li>
    </ol>
    <p id="err"></p>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const DATA = __DATA__;
(function(){
if(!window.THREE){document.getElementById('err').textContent='Không tải được thư viện three.js (cần mạng lần đầu mở trang).';return;}
const KIND_COLOR={servo:0x2b2f36,bracket:0xb9bec5,link:0x8f98a3,gear:0xf2efe6,penbox:0x3f7fbf,pen:0xc8372d};
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const canvas=document.getElementById('gl'), stage=document.getElementById('stage');
const renderer=new THREE.WebGLRenderer({canvas,antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(32,1,0.01,10);
camera.up.set(0,0,1);
const controls=new THREE.OrbitControls(camera,canvas);
const CENTER=new THREE.Vector3(-0.015,-0.06,0.27);
scene.add(new THREE.HemisphereLight(0xffffff,0x8a8f98,0.85));
const sun=new THREE.DirectionalLight(0xffffff,0.75); scene.add(sun);

function parseSTL(b64){
  const bin=atob(b64), n=bin.length, u8=new Uint8Array(n);
  for(let i=0;i<n;i++)u8[i]=bin.charCodeAt(i);
  const dv=new DataView(u8.buffer), tris=dv.getUint32(80,true), pos=new Float32Array(tris*9);
  for(let t=0;t<tris;t++){const o=84+50*t+12; for(let k=0;k<9;k++)pos[t*9+k]=dv.getFloat32(o+4*k,true);}
  const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.BufferAttribute(pos,3)); g.computeVertexNormals(); return g;
}

// ── cây khớp ──
const linkGroup={}, meshes=[], jointRot={}, jointOrigin={};
DATA.links.forEach(L=>{
  const g=new THREE.Group(); linkGroup[L.name]=g;
  const mat=new THREE.MeshStandardMaterial({color:KIND_COLOR[L.kind],roughness:0.62,metalness:L.kind==='bracket'||L.kind==='link'?0.35:0.05,transparent:true});
  const m=new THREE.Mesh(parseSTL(L.stl),mat);
  m.matrixAutoUpdate=false;
  m.matrix.set(...L.vis).multiply(new THREE.Matrix4().makeScale(0.001,0.001,0.001));
  m.userData=L; g.add(m); meshes.push(m);
});
scene.add(linkGroup.base_link);
DATA.joints.forEach(J=>{
  const o=new THREE.Group(); o.position.set(...J.xyz); linkGroup[J.parent].add(o);
  let holder=o;
  if(J.type==='revolute'){const r=new THREE.Group(); o.add(r); holder=r; jointRot[J.name]=r; jointOrigin[J.name]=o; r.userData.axis=new THREE.Vector3(...J.axis);}
  holder.add(linkGroup[J.child]);
});
const tipObj=new THREE.Mesh(new THREE.SphereGeometry(0.0022,16,12),new THREE.MeshBasicMaterial({color:0xff7a1a}));
tipObj.position.set(...DATA.tool); linkGroup.hopbut_1.add(tipObj);

// ── trục khớp + mũi tên chiều dương ──
const axesGroup=[];
const ACC=new THREE.Color(0xe2622b);
Object.keys(jointOrigin).forEach(name=>{
  const a=jointRot[name].userData.axis.clone().normalize(), g=new THREE.Group();
  const u=Math.abs(a.z)>0.9?new THREE.Vector3(1,0,0):new THREE.Vector3(0,0,1); u.sub(a.clone().multiplyScalar(u.dot(a))).normalize();
  const v=a.clone().cross(u), R=0.03, pts=[];
  for(let i=0;i<=48;i++){const t=i/48*1.5*Math.PI; pts.push(u.clone().multiplyScalar(R*Math.cos(t)).add(v.clone().multiplyScalar(R*Math.sin(t))));}
  const lm=new THREE.LineBasicMaterial({color:ACC,depthTest:false});
  g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),lm));
  g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([a.clone().multiplyScalar(-0.05),a.clone().multiplyScalar(0.05)]),
        new THREE.LineDashedMaterial({color:ACC,dashSize:0.006,gapSize:0.004,depthTest:false})));
  g.children[1].computeLineDistances();
  const te=1.5*Math.PI, end=pts[pts.length-1], tan=u.clone().multiplyScalar(-Math.sin(te)).add(v.clone().multiplyScalar(Math.cos(te)));
  const cone=new THREE.Mesh(new THREE.ConeGeometry(0.004,0.012,12),new THREE.MeshBasicMaterial({color:ACC,depthTest:false}));
  cone.position.copy(end); cone.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),tan); g.add(cone);
  g.renderOrder=5; g.children.forEach(c=>c.renderOrder=5);
  jointOrigin[name].add(g); axesGroup.push(g);
});

// ── nhãn HTML chiếu từ 3D ──
const labels=[];
function addLabel(cls,text,posFn){const el=document.createElement('div'); el.className='lbl '+cls; el.textContent=text; stage.appendChild(el); const l={el,posFn,show:true}; labels.push(l); return l;}
const jLabels=Object.keys(jointOrigin).map(name=>{const i=DATA.jointInfo[name];
  return addLabel('j',i[0]+' '+name+' · kênh '+i[2],()=>{const a=jointRot[name].userData.axis; return jointOrigin[name].localToWorld(Math.abs(a.z)>0.9?new THREE.Vector3(0.075,0,name==='base'?0.03:0):a.clone().multiplyScalar(0.07));});});

// ── đường kích thước trong mặt phẳng tay ──
const dimMat=new THREE.LineBasicMaterial({color:0x1f6fd0,depthTest:false});
const dimGeo=new THREE.BufferGeometry(); dimGeo.setAttribute('position',new THREE.BufferAttribute(new Float32Array(12),3));
const dimLine=new THREE.Line(dimGeo,dimMat); dimLine.renderOrder=4; dimLine.frustumCulled=false; scene.add(dimLine);
const dimPts=[new THREE.Vector3(),new THREE.Vector3(),new THREE.Vector3(),new THREE.Vector3()];
const dimNames=['J2→J3','J3→J4','J4→đầu bút'];
const dLabels=dimNames.map((n,i)=>addLabel('d','',()=>dimPts[i].clone().add(dimPts[i+1]).multiplyScalar(0.5)));
function updateDims(){
  const ref=jointOrigin.shoulder, src=[jointOrigin.shoulder,jointOrigin.elbow,jointOrigin.wrist_roll,tipObj];
  src.forEach((o,i)=>{const p=o.getWorldPosition(new THREE.Vector3()); ref.worldToLocal(p); p.x=0; ref.localToWorld(p); dimPts[i].copy(p); dimGeo.attributes.position.setXYZ(i,p.x,p.y,p.z);});
  dimGeo.attributes.position.needsUpdate=true;
  dLabels.forEach((l,i)=>l.el.textContent=dimNames[i]+' '+(dimPts[i].distanceTo(dimPts[i+1])*1000).toFixed(1));
}

// ── điều khiển khớp ──
const q={base:0,shoulder:0,elbow:0,wrist_roll:0};
let elbowHome=30;
const jBox=document.getElementById('joints'), ui={};
function limits(name){ if(name==='elbow') return elbowHome===30?[-30,150]:[-90,90]; return [-90,90]; }
Object.keys(DATA.jointInfo).forEach(name=>{
  const i=DATA.jointInfo[name], row=document.createElement('div'); row.className='jrow';
  row.innerHTML='<label>'+i[0]+' · '+name+' <small>'+i[1]+', kênh '+i[2]+'</small></label><output></output><input type="range" step="1"><div class="servo"></div>';
  jBox.appendChild(row);
  const inp=row.querySelector('input'); ui[name]={inp,out:row.querySelector('output'),servo:row.querySelector('.servo')};
  inp.addEventListener('input',()=>{q[name]=+inp.value; apply();});
});
function apply(){
  Object.keys(q).forEach(name=>{
    const [lo,hi]=limits(name); q[name]=Math.max(lo,Math.min(hi,q[name]));
    const u=ui[name]; u.inp.min=lo; u.inp.max=hi; u.inp.value=q[name];
    u.out.textContent=(q[name]>0?'+':'')+q[name]+'°';
    const home=name==='elbow'?elbowHome:90;
    u.servo.textContent='lệnh servo '+(home+q[name])+' (home '+home+')';
    jointRot[name].setRotationFromAxisAngle(jointRot[name].userData.axis,q[name]*Math.PI/180);
  });
  scene.updateMatrixWorld(true); updateDims();
  const p=tipObj.getWorldPosition(new THREE.Vector3());
  document.getElementById('tipRead').innerHTML='Đầu bút (base_link, mm): <b>x '+(p.x*1000).toFixed(1)+' · y '+(p.y*1000).toFixed(1)+' · z '+(p.z*1000).toFixed(1)+'</b><br>Bút nghiêng khỏi phương đứng: <b>'+(q.shoulder+q.elbow)+'°</b>';
}
function pose(a){Object.assign(q,{base:a[0],shoulder:a[1],elbow:a[2],wrist_roll:a[3]}); apply();}
document.getElementById('bHome').onclick=()=>pose([0,0,0,0]);
document.getElementById('bCad').onclick=()=>pose([0,90,0,0]);
document.getElementById('bDraw').onclick=()=>pose(elbowHome===30?[-12,1,102,0]:[0,35,70,0]);
document.getElementById('elbowWin').onchange=e=>{elbowHome=+e.target.value; apply();};

// ── danh sách chi tiết, chọn, làm mờ ──
let pinned=null, hovered=null;
const pBox=document.getElementById('parts'), pRow={};
meshes.forEach(m=>{
  const L=m.userData, row=document.createElement('label');
  row.innerHTML='<input type="checkbox" checked><span class="sw" style="background:#'+KIND_COLOR[L.kind].toString(16).padStart(6,'0')+'"></span><span>'+L.name+'</span>';
  row.querySelector('input').addEventListener('change',e=>{m.visible=e.target.checked;});
  row.addEventListener('mouseenter',()=>setHover(m)); row.addEventListener('mouseleave',()=>setHover(null));
  row.querySelector('span:last-child').addEventListener('click',e=>{e.preventDefault(); pin(m);});
  pBox.appendChild(row); pRow[L.name]=row;
});
function refresh(){
  const cur=hovered||pinned, ghost=document.getElementById('tGhost').checked&&pinned;
  meshes.forEach(m=>{
    m.material.emissive.setHex(m===pinned?0x1f4f9a:(m===hovered?0x7a3410:0x000000));
    m.material.opacity=ghost&&m!==pinned?0.12:1; m.material.depthWrite=!(ghost&&m!==pinned);
    pRow[m.userData.name].classList.toggle('on',m===pinned);
  });
  document.getElementById('tipName').textContent=cur?cur.userData.name:'—';
  document.getElementById('tipDesc').textContent=cur?cur.userData.desc:'Rê chuột lên một chi tiết, hoặc bấm để ghim.';
}
function setHover(m){hovered=m; refresh();}
function pin(m){pinned=(pinned===m?null:m); refresh();}
const ray=new THREE.Raycaster(), ptr=new THREE.Vector2();
function pick(ev){const r=canvas.getBoundingClientRect(); ptr.set((ev.clientX-r.left)/r.width*2-1,-(ev.clientY-r.top)/r.height*2+1);
  ray.setFromCamera(ptr,camera); const h=ray.intersectObjects(meshes.filter(m=>m.visible)); return h.length?h[0].object:null;}
let downAt=null;
canvas.addEventListener('pointermove',ev=>{if(ev.buttons===0)setHover(pick(ev));});
canvas.addEventListener('pointerleave',()=>setHover(null));
canvas.addEventListener('pointerdown',ev=>{downAt=[ev.clientX,ev.clientY];});
canvas.addEventListener('pointerup',ev=>{if(downAt&&Math.hypot(ev.clientX-downAt[0],ev.clientY-downAt[1])<4){const m=pick(ev); if(m)pin(m); else {pinned=null;refresh();}}});
document.getElementById('tGhost').onchange=refresh;
document.getElementById('tAxes').onchange=e=>{axesGroup.forEach(g=>g.visible=e.target.checked); jLabels.forEach(l=>l.show=e.target.checked);};
document.getElementById('tDims').onchange=e=>{dimLine.visible=e.target.checked; dLabels.forEach(l=>l.show=e.target.checked);};

// ── góc nhìn ──
const VIEWS={iso:[0.62,-0.72,0.18],front:[0,-1,0.02],side:[1,0,0.02],top:[0.001,-0.02,1]};
function setView(k){const d=new THREE.Vector3(...VIEWS[k]).normalize().multiplyScalar(0.88); controls.target.copy(CENTER); camera.position.copy(CENTER).add(d); controls.update();}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>setView(b.dataset.view));

function resize(){const w=canvas.clientWidth,h=canvas.clientHeight; if(canvas.width!==Math.round(w*renderer.getPixelRatio())||canvas.height!==Math.round(h*renderer.getPixelRatio())){renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix();}}
const v3=new THREE.Vector3();
function frame(){
  resize(); controls.update();
  renderer.setClearColor(new THREE.Color(css('--stage')||'#e9ecef'));
  sun.position.copy(camera.position).add(new THREE.Vector3(0.2,0.1,0.5));
  renderer.render(scene,camera);
  const w=canvas.clientWidth,h=canvas.clientHeight;
  labels.forEach(l=>{v3.copy(l.posFn()).project(camera); const ok=l.show&&v3.z<1&&Math.abs(v3.x)<1&&Math.abs(v3.y)<1;
    l.el.hidden=!ok; if(ok){l.el.style.left=((v3.x+1)/2*w)+'px'; l.el.style.top=((1-v3.y)/2*h)+'px';}});
  requestAnimationFrame(frame);
}
setView('iso'); pose([0,0,0,0]); refresh(); frame();
})();
</script>
"""


def build():
    links, joints = S.load_export()
    T_cad = S.link_poses(joints, S.Q_CAD)
    mesh_dir = S.PKG / "meshes" / "newarm"
    out_links = []
    for name, L in links.items():
        stl = mesh_dir / ("base_link.stl" if name == "base_link" else L["mesh"])
        kind, desc = PARTS[name]
        out_links.append({
            "name": name, "kind": kind, "desc": desc,
            "vis": [round(float(v), 9) for v in np.linalg.inv(T_cad[name]).flatten()],
            "stl": base64.b64encode(stl.read_bytes()).decode(),
        })
    out_joints = [{"name": j["name"], "type": j["type"], "parent": j["parent"], "child": j["child"],
                   "xyz": [float(v) for v in j["xyz"]], "axis": j["axis"]} for j in joints]
    data = {"links": out_links, "joints": out_joints, "tool": list(F.TOOL_OFFSET), "jointInfo": JOINT_INFO}
    return PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body-only", metavar="FILE", help="ghi bản không có <!doctype> (để publish)")
    args = ap.parse_args()
    page = build()
    if args.body_only:
        Path(args.body_only).write_text(page)
        print(f"{args.body_only}: {len(page)/1e6:.1f} MB")
    out = S.REPO / "docs" / "newarm_cad_viewer.html"
    out.write_text('<!doctype html>\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width,initial-scale=1">\n' + page)
    print(f"{out}: {out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
