// main.js — wires the whole dashboard together: robot simulation viewport,
// two anatomical CNS views (one shared connectome scene, two cameras), the
// joint timeline, HUD readouts and the toggle chips.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { buildRobot } from './robot.js';
import { GaitController, JOINT_ORDER } from './gait.js';
import { Connectome, NEURON_COUNT, CONNECTION_COUNT } from './connectome.js';
import { Timeline } from './timeline.js';
import { createDuckPhysics, mjcfToWorld, DEFAULT_POSE, OBSTACLES, MAZE_WALLS } from './physics.js';

const LOOP_S = 20;

// ---------- boot ----------
const state = {
  playing: true,
  chips: { wireframe: false, connectome: false, controller: true, physics: false, ocean: true },
};

const robot = await buildRobot('assets/robot.json', 'assets/meshes/');
robot.root.rotation.set(-Math.PI / 2, 0, 0, 'YXZ'); // MJCF z-up → three y-up

const gait = new GaitController(robot);
const cns = new Connectome();
window.__dbg = { gait, robot, cns, THREE };

// Real MuJoCo physics + trained RL policies (stand/walk ONNX actors). Falls
// back to the procedural gait if the runtimes cannot load (offline CDN).
let physics = null;
let physicsFailed = false;
const loadBadge = document.getElementById('loadBadge');
createDuckPhysics({
  onProgress: (m) => {
    loadBadge.textContent = 'RL PHYSICS · ' + m.toUpperCase();
    loadBadge.style.display = 'block';
  },
}).then((p) => {
  physics = p;
  window.__physics = p;
  if (state.chips.controller && state.playing) p.resume(); else p.pause();
  p.start();
  loadBadge.style.display = 'none';
}).catch((err) => {
  console.error('RL physics init failed — procedural fallback', err);
  physicsFailed = true;
  gait.trained = state.chips.controller;
  loadBadge.textContent = 'RL RUNTIME UNAVAILABLE · PROCEDURAL FALLBACK';
  setTimeout(() => { loadBadge.style.display = 'none'; }, 5000);
});
const timeline = new Timeline(document.getElementById('tracks'));
timeline.setRanges(robot.joints);

document.getElementById('statNeurons').textContent = NEURON_COUNT;
document.getElementById('statConns').textContent = CONNECTION_COUNT.toLocaleString('en-US');

// ---------- simulation scene ----------
const simCanvas = document.getElementById('simCanvas');
const simScene = new THREE.Scene();
const SKY = { ocean: 0x0e1726, void: 0x05070b };
simScene.background = new THREE.Color(SKY.ocean);
simScene.fog = new THREE.Fog(SKY.ocean, 1.6, 5.2);

const simCam = new THREE.PerspectiveCamera(38, 1, 0.01, 60);
simCam.position.set(0.35, 0.85, 0.75); // close 3/4 chase view, duck fills the frame
const orbit = new OrbitControls(simCam, simCanvas);
orbit.target.set(0, 0.075, 0); // spawn point — followRobot keeps the offset from there
orbit.enableDamping = true;
orbit.minDistance = 0.15;
orbit.maxDistance = 8;
orbit.update();

// camera rig follows the walking robot while preserving user orbit offsets
const followPrev = new THREE.Vector3();
function followRobot() {
  const p = new THREE.Vector3();
  robot.trunk.getWorldPosition(p);
  const delta = p.clone().sub(followPrev);
  if (followPrev.lengthSq() > 0) {
    simCam.position.add(delta);
    orbit.target.add(delta);
  }
  followPrev.copy(p);
  orbit.target.y = 0.075;
}

// ocean-floor checkerboard
function checkerTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d');
  g.fillStyle = '#1a2635';
  g.fillRect(0, 0, 256, 256);
  g.fillStyle = '#202f42';
  g.fillRect(0, 0, 128, 128);
  g.fillRect(128, 128, 128, 128);
  g.strokeStyle = 'rgba(110,150,190,0.35)';
  g.lineWidth = 3;
  g.strokeRect(0, 0, 256, 256);
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(42, 42);
  tex.anisotropy = 8;
  return tex;
}
const floorMat = new THREE.MeshStandardMaterial({ map: checkerTexture(), roughness: 0.95, metalness: 0.0 });
const floor = new THREE.Mesh(new THREE.PlaneGeometry(28, 28), floorMat);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
simScene.add(floor);

simScene.add(new THREE.AmbientLight(0xbfd4e8, 0.55));
const key = new THREE.SpotLight(0xffffff, 26, 0, 0.55, 0.85, 1.4);
key.position.set(0.9, 2.2, 0.7);
key.castShadow = true;
key.shadow.mapSize.set(1024, 1024);
key.target.position.set(0, 0, 0);
simScene.add(key, key.target);
const fill = new THREE.DirectionalLight(0x88aacc, 0.85);
fill.position.set(-1, 1.4, -0.6);
simScene.add(fill);
const rim = new THREE.DirectionalLight(0x56d9d3, 0.6);
rim.position.set(-0.4, 0.5, 1);
simScene.add(rim);

simScene.add(robot.root);

// ---------- obstacles ----------
// Maze walls + the rocks beyond the exit. Both arrays are the exact geoms
// physics.js injects into MuJoCo (MuJoCo z-up half sizes) — what the whisker
// rays see is exactly what the body can hit.
const rockMat = new THREE.MeshStandardMaterial({
  color: 0x2b3849, roughness: 0.9, metalness: 0.05, side: THREE.DoubleSide,
});
const wallMat = new THREE.MeshStandardMaterial({
  color: 0x41536b, roughness: 0.85, metalness: 0.05, side: THREE.DoubleSide,
});
const rockMeshes = [];
for (const [arr, mat] of [[MAZE_WALLS, wallMat], [OBSTACLES, rockMat]]) {
  for (const { pos, size } of arr) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(size[0] * 2, size[2] * 2, size[1] * 2), mat);
    m.position.set(pos[0], pos[2], -pos[1]); // mjcfToWorld convention: (x, z, -y)
    m.castShadow = m.receiveShadow = true;
    simScene.add(m);
    rockMeshes.push(m);
  }
}

// ---------- vision: head-height whisker rays → visL/visR closeness ----------
const VISION_FAR = 1.3, VISION_NEAR = 0.25;
const RAY_YAW = [-0.95, -0.5, 0, 0.5, 0.95]; // rad, body frame; + = left side
const raycaster = new THREE.Raycaster();
raycaster.far = VISION_FAR;
const _rayO = new THREE.Vector3(), _fwd = new THREE.Vector3();
const _left = new THREE.Vector3(), _dir = new THREE.Vector3(), _end = new THREE.Vector3();
const VISION_UP = new THREE.Vector3(0, 1, 0);
const rayGeo = new THREE.BufferGeometry();
rayGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(RAY_YAW.length * 6), 3));
rayGeo.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(RAY_YAW.length * 6), 3));
const rayLines = new THREE.LineSegments(rayGeo, new THREE.LineBasicMaterial({
  vertexColors: true, transparent: true, opacity: 0.45,
  blending: THREE.AdditiveBlending, depthWrite: false,
}));
simScene.add(rayLines);

function updateVision() {
  robot.trunk.getWorldPosition(_rayO);
  _rayO.y += 0.10; // head-ish height, kept below the rock tops (≥0.32 m)
  _fwd.set(1, 0, 0).applyQuaternion(robot.root.quaternion); // MuJoCo body +x
  _left.crossVectors(VISION_UP, _fwd).normalize();
  let visL = 0, visR = 0;
  const pos = rayGeo.attributes.position, col = rayGeo.attributes.color;
  for (let k = 0; k < RAY_YAW.length; k++) {
    const th = RAY_YAW[k];
    _dir.copy(_fwd).multiplyScalar(Math.cos(th)).addScaledVector(_left, Math.sin(th)).normalize();
    raycaster.set(_rayO, _dir);
    const hits = raycaster.intersectObjects(rockMeshes, false);
    const d = hits.length ? hits[0].distance : VISION_FAR;
    const inten = THREE.MathUtils.clamp((VISION_FAR - d) / (VISION_FAR - VISION_NEAR), 0, 1);
    if (th >= 0) visL = Math.max(visL, inten); else visR = Math.max(visR, inten);
    if (hits.length) _end.copy(hits[0].point); else _end.copy(_rayO).addScaledVector(_dir, VISION_FAR);
    pos.setXYZ(2 * k, _rayO.x, _rayO.y, _rayO.z);
    pos.setXYZ(2 * k + 1, _end.x, _end.y, _end.z);
    const r = hits.length ? 1.0 : 0.16, g = hits.length ? 0.55 : 0.75, b = hits.length ? 0.25 : 0.8;
    col.setXYZ(2 * k, r, g, b);
    col.setXYZ(2 * k + 1, r, g, b);
  }
  pos.needsUpdate = true;
  col.needsUpdate = true;
  return { visL, visR };
}

// ---------- CNS scenes ----------
const cnsScene = new THREE.Scene();
cnsScene.background = new THREE.Color('#070b12');
cnsScene.add(cns.group);

const cnsFullCanvas = document.getElementById('cnsFullCanvas');
const cnsVncCanvas = document.getElementById('cnsVncCanvas');
const cnsFullCam = new THREE.PerspectiveCamera(30, 1, 10, 8000);
cnsFullCam.position.set(0, -60, 2720);
cnsFullCam.lookAt(20, -70, 0);
// VNC detail — truly zoomed in on the ventral-nerve-cord motor pools
// (FOV 15° + distance ~700 yields a vertical extent of ~184 world units, so
//  the view frames the meso+meta thoracic pools (~y∈[-215,-100]) tightly
//  and excludes the brain / descending populations above)
const cnsVncCam = new THREE.PerspectiveCamera(15, 1, 10, 8000);
cnsVncCam.position.set(-220, -210, 700);
cnsVncCam.lookAt(0, -210, 0);

// ---------- renderers ----------
// try/catch around WebGLRenderer — headless / GPU-blocked / sandboxed browsers
// throw here. We return null and the frame loop still ticks (CNS integration,
// timeline, HUD all keep working); only the 3D viewport is dark.
let lastWebGLError = null;
function makeRenderer(canvas) {
  try {
    const r = new THREE.WebGLRenderer({ canvas, antialias: true });
    r.setPixelRatio(Math.min(devicePixelRatio, 2));
    r.shadowMap.enabled = canvas === simCanvas;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    return r;
  } catch (err) {
    console.error('[render] WebGL unavailable for', canvas?.id, err);
    lastWebGLError = err;
    return null;
  }
}
const simR = makeRenderer(simCanvas);
const fullR = makeRenderer(cnsFullCanvas);
const vncR = makeRenderer(cnsVncCanvas);

// Friendly overlay on any canvas whose WebGL context failed to create — the
// CNS brain keeps integrating underneath, the HUD / timeline still update, so
// tell the user exactly that and how to recover.
function webglFallback(canvas, label) {
  const el = document.createElement('div');
  el.className = 'webgl-fallback';
  const detail = (lastWebGLError?.message || '').replace(/[<>&]/g, ' ').slice(0, 160);
  el.innerHTML = `
    <div class="wf-title">WEBGL UNAVAILABLE</div>
    <div class="wf-msg">3D 视图 (${label}) 无法渲染。<br>
      <b>脑动力学仍在后台运行</b> — 查看底部 HUD 与时间轴可观察活动。</div>
    <div class="wf-hint">TRY A HARDWARE-ACCELERATED BROWSER</div>
    ${detail ? `<div class="wf-detail">${detail}</div>` : ''}
  `;
  canvas.parentElement.appendChild(el);
}
if (!simR) webglFallback(simCanvas, 'robot simulation');
if (!fullR) webglFallback(cnsFullCanvas, 'full CNS context');
if (!vncR) webglFallback(cnsVncCanvas, 'VNC detail');
if (!simR || !fullR || !vncR) {
  console.warn('[render] one or more WebGL contexts failed — CNS integration still runs in the main loop');
}

function fitRenderer(r, canvas, cam) {
  const box = canvas.parentElement.getBoundingClientRect();
  const w = Math.max(2, box.width), h = Math.max(2, box.height);
  if (r) r.setSize(w, h, false);
  cam.aspect = w / h;
  cam.updateProjectionMatrix();
}
function fitAll() {
  fitRenderer(simR, simCanvas, simCam);
  fitRenderer(fullR, cnsFullCanvas, cnsFullCam);
  fitRenderer(vncR, cnsVncCanvas, cnsVncCam);
  timeline.draw();
}
new ResizeObserver(fitAll).observe(document.getElementById('app'));
fitAll();

// ---------- UI state ----------
document.querySelectorAll('.chip').forEach((btn) => {
  btn.addEventListener('click', () => {
    const k = btn.dataset.chip;
    state.chips[k] = !state.chips[k];
    btn.classList.toggle('active', state.chips[k]);
    applyChip(k);
  });
});

function applyChip(k) {
  const on = state.chips[k];
  switch (k) {
    case 'wireframe': robot.setAllWireframe(on); break;
    case 'connectome': cns.setDynamics(on); break;
    case 'controller':
      // ON = real RL policy via MuJoCo; OFF = untrained procedural ablation.
      if (physics) { on ? physics.resume() : physics.pause(); }
      gait.trained = physicsFailed ? on : false;
      break;
    case 'physics': gait.physics = on; break;
    case 'cinema':
      // hide every overlay — nothing left but the duck and the orbit camera
      document.body.classList.toggle('cinema', on);
      break;
    case 'ocean': {
      simScene.background.set(on ? SKY.ocean : SKY.void);
      if (simScene.fog) simScene.fog.color.set(on ? SKY.ocean : SKY.void);
      simScene.fog = on ? new THREE.Fog(SKY.ocean, 1.6, 5.2) : null;
      floorMat.map = on ? checkerTexture() : null;
      floorMat.color.set(on ? 0xffffff : 0x0a0e14);
      floorMat.needsUpdate = true;
      break;
    }
  }
}
applyChip('controller'); // sync initial trained state
state.chips.controller = true;
document.querySelector('[data-chip="controller"]').classList.add('active');

// overlay toggles
document.getElementById('btnTracks')?.addEventListener('click', () => {
  document.getElementById('tracksDock').classList.toggle('open');
  timeline.draw();
});
document.getElementById('cnsTab')?.addEventListener('click', () => {
  const panel = document.getElementById('cnsPanel');
  const closed = panel.classList.toggle('closed');
  document.getElementById('cnsTab').textContent = closed ? '⟨' : '⟩';
});
document.getElementById('cinemaExit')?.addEventListener('click', () => {
  document.querySelector('[data-chip="cinema"]')?.click();
});

// transport
const btnPlay = document.getElementById('btnPlay');
btnPlay.textContent = '❚❚';
btnPlay.addEventListener('click', () => {
  state.playing = !state.playing;
  btnPlay.textContent = state.playing ? '❚❚' : '▶';
});
document.getElementById('btnFs').addEventListener('click', () => {
  const app = document.getElementById('app');
  if (document.fullscreenElement) document.exitFullscreen();
  else app.requestFullscreen?.();
});
document.getElementById('btnMute').addEventListener('click', (e) => {
  const b = e.currentTarget;
  b.textContent = b.textContent === '🔊' ? '🔇' : '🔊';
});

// scrub
const scrub = document.getElementById('scrub');
const scrubPlayed = document.getElementById('scrubPlayed');
let scrubFrac = null;
function scrubFromEvent(e) {
  const box = scrub.getBoundingClientRect();
  const x = (e.touches ? e.touches[0].clientX : e.clientX) - box.left;
  return THREE.MathUtils.clamp(x / box.width, 0, 1);
}
scrub.addEventListener('pointerdown', (e) => {
  scrubFrac = scrubFromEvent(e);
  scrub.setPointerCapture(e.pointerId);
  state.playing = false;
  btnPlay.textContent = '❚❚';
});
scrub.addEventListener('pointermove', (e) => { if (scrubFrac !== null) scrubFrac = scrubFromEvent(e); });
scrub.addEventListener('pointerup', () => { scrubFrac = null; });

// ---------- HUD ----------
const hudTime = document.getElementById('hudTime');
const hudDuration = document.getElementById('hudDuration');
const hudBody = document.getElementById('hudBody');
const hudUnit = document.getElementById('hudUnit');
const hudState = document.getElementById('hudState');
const hudPolicy = document.getElementById('hudPolicy');
const hudMode = document.getElementById('hudMode');
const tcNow = document.getElementById('tcNow');

function fmtTime(t) {
  const s = Math.floor(t);
  return `0:${String(s).padStart(2, '0')}`;
}

// ---------- main loop ----------
const clock = new THREE.Clock();
let simTime = 0;
let prevHeading = gait.heading;
let headingRate = 0;
const replayAngles = {}; const replayDrives = {};
JOINT_ORDER.forEach((j) => { replayAngles[j] = 0; replayDrives[j] = 0; });
const physDrives = {}; const physPrev = {};
JOINT_ORDER.forEach((j) => { physDrives[j] = 0; physPrev[j] = 0; });
const _eul = new THREE.Euler();
const _lightP = new THREE.Vector3();

function physicsSnapshot(st, dt) {
  // drive the visual rig from the MuJoCo state and build a snapshot with the
  // same shape the procedural gait produces
  mjcfToWorld(st.pos, st.quat, robot.root.position, robot.root.quaternion);
  const angles = {}; const drives = {};
  for (let j = 0; j < JOINT_ORDER.length; j++) {
    const name = JOINT_ORDER[j];
    robot.setJoint(name, st.angles[j]);
    angles[name] = st.angles[j];
    const prev = physPrev[name];
    const range = robot.joints.get(name).range;
    const scale = Math.max(1e-4, (range[1] - range[0]) / 2);
    const vel = (st.angles[j] - prev) / Math.max(dt, 1e-3);
    const dVel = THREE.MathUtils.clamp((vel / scale) * 0.35, -1, 1);
    const dAct = THREE.MathUtils.clamp(st.action[j], -1, 1);
    const d = THREE.MathUtils.clamp(0.6 * dVel + 0.4 * dAct, -1, 1);
    physDrives[name] += (d - physDrives[name]) * Math.min(1, dt * 10);
    drives[name] = physDrives[name];
    physPrev[name] = st.angles[j];
  }
  _eul.setFromQuaternion(robot.root.quaternion, 'YXZ');
  return {
    t: st.time,
    angles, drives,
    body: {
      heading: -_eul.y, roll: _eul.z, pitch: _eul.x,
      height: robot.root.position.y,
      speed: st.speed,
      pos: robot.root.position.clone(),
    },
    scanYaw: angles['head_yaw'] ?? 0,
    eventDuration: physics.modeSinceSteps * 0.02,
  };
}

function holdStandSnapshot(dt) {
  // physics runtimes still loading: hold the STAND keyframe pose kinematically
  for (let j = 0; j < JOINT_ORDER.length; j++) robot.setJoint(JOINT_ORDER[j], DEFAULT_POSE[j]);
  robot.root.position.set(0, -0.012, 0);
  robot.root.rotation.set(-Math.PI / 2, 0, 0, 'YXZ');
  const angles = {}; const drives = {};
  JOINT_ORDER.forEach((j) => { angles[j] = DEFAULT_POSE[j]; drives[j] = 0; });
  simTime = (simTime + dt) % LOOP_S;
  return {
    t: simTime, angles, drives,
    body: { heading: 0, roll: 0, pitch: 0, height: -0.012, speed: 0, pos: new THREE.Vector3() },
    scanYaw: 0, eventDuration: 0,
  };
}

function frame() {
  requestAnimationFrame(frame);
  try {
  const dt = Math.min(clock.getDelta(), 0.05);

  let snap = null;
  let sensory = { speed: 0, yawRate: 0, upright: 1, fallen: 0, headSpeed: 0 };
  const useRL = physics && state.chips.controller && !physicsFailed;
  if (useRL) {
    if (state.playing && scrubFrac === null) physics.resume(); else physics.pause();
    const st = physics.getState();
    snap = physicsSnapshot(st, dt);
    // proprioceptive + vestibular channels feeding the connectome brain
    const upr = THREE.MathUtils.clamp((-st.gravityZ - 0.55) / 0.3, 0, 1);
    sensory = { speed: st.speed, yawRate: st.yawRate, upright: upr, fallen: 1 - upr, headSpeed: st.headSpeed };
    simTime = snap.t % LOOP_S;
    timeline.record(snap.angles, snap.drives);
  } else if (!physics && state.chips.controller && !physicsFailed) {
    snap = holdStandSnapshot(dt);
    prevHeading = 0; headingRate = 0;
  } else if (state.playing) {
    snap = gait.update(dt);
    simTime = (simTime + dt) % LOOP_S;
    timeline.record(snap.angles, snap.drives);
    sensory = { speed: snap.body.speed, yawRate: headingRate, upright: 1, fallen: 0, headSpeed: 0 };
  } else if (scrubFrac !== null) {
    timeline.sampleAt(scrubFrac, replayAngles, replayDrives);
    for (const j of JOINT_ORDER) robot.setJoint(j, replayAngles[j]);
    if (physics) {
      const st = physics.getState();
      mjcfToWorld(st.pos, st.quat, robot.root.position, robot.root.quaternion);
      snap = physicsSnapshot(st, dt);
      snap.angles = { ...replayAngles }; snap.drives = { ...replayDrives };
      for (const j of JOINT_ORDER) robot.setJoint(j, replayAngles[j]);
    } else {
      snap = gait.snapshot();
      snap.drives = { ...replayDrives };
    }
  } else {
    snap = gait.snapshot();
  }
  {
    const hr = (snap.body.heading - prevHeading) / Math.max(dt, 1e-4);
    headingRate += (hr - headingRate) * Math.min(1, dt * 4);
    prevHeading = snap.body.heading;
  }

  // whisker rays sample the world with the body pose that was just written,
  // then the closeness signal rides into the brain alongside proprioception
  const vis = updateVision();
  window.__dbg.vision = vis;
  sensory.visL = vis.visL;
  sensory.visR = vis.visR;

  // the brain steps the full CNS dynamics and decodes commands from its
  // descending population; the RL policy executes them (balance + gait)
  const cmds = cns.update({ drives: snap.drives, sensory }, dt);
  if (useRL) {
    physics.setCommand(cmds.vx, 0, cmds.wz);
    physics.setHeadOffsets(cmds.head);
  }

  // HUD
  const epT = simTime;
  hudTime.textContent = epT.toFixed(2).padStart(5, '0') + 's';
  hudDuration.textContent = snap.eventDuration.toFixed(2) + 's';
  const wrap180 = (deg) => ((deg % 360) + 540) % 360 - 180;
  hudBody.textContent =
    `${Math.round(wrap180(snap.body.heading * 180 / Math.PI) * 10) / 10}/${Math.round(wrap180(snap.body.pitch * 180 / Math.PI) * 10) / 10}`;
  // most active descending unit = current "model unit"
  let best = 0, bestA = -1;
  for (let i = 56; i < 72; i++) {
    const a = Math.abs(cns.activity[i]);
    if (a > bestA) { bestA = a; best = i; }
  }
  hudUnit.textContent = best;
  hudState.textContent = bestA.toFixed(3);
  hudPolicy.textContent = physics ? physics.policy : (physicsFailed ? 'procedural' : 'loading');
  hudMode.textContent = physics ? physics.mode : '—';
  tcNow.textContent = fmtTime(epT);
  scrubPlayed.style.width = `${(epT / LOOP_S) * 100}%`;

  if (state.playing) timeline.draw();

  followRobot();
  orbit.update();
  // fog breathes with zoom: moody haze up close, clear view of the whole maze
  // when the camera pulls back (fixed fog made zoomed-out views pitch black)
  const camDist = simCam.position.distanceTo(orbit.target);
  if (simScene.fog) {
    simScene.fog.near = 1.6 + camDist * 0.6;
    simScene.fog.far = 5.2 + camDist * 2.6;
  }
  // the key spotlight pools on the robot wherever it roams the maze
  robot.trunk.getWorldPosition(_lightP);
  key.position.set(_lightP.x + 0.9, _lightP.y + 2.2, _lightP.z + 0.7);
  key.target.position.set(_lightP.x, 0, _lightP.z);
  simR?.render(simScene, simCam);
  fullR?.render(cnsScene, cnsFullCam);
  vncR?.render(cnsScene, cnsVncCam);
  } catch (err) {
    // never let a single bad frame kill the loop — log and keep ticking so
    // CNS integration, timeline and HUD all stay alive
    console.error('[frame] loop error (will keep ticking):', err);
  }
}
frame();
