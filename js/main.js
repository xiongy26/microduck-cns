// main.js — wires the whole dashboard together: robot simulation viewport,
// two anatomical CNS views (one shared connectome scene, two cameras), the
// joint timeline, HUD readouts and the toggle chips.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { buildRobot } from './robot.js';
import { GaitController, JOINT_ORDER } from './gait.js';
import { Connectome, NEURON_COUNT, CONNECTION_COUNT } from './connectome.js';
import { Timeline } from './timeline.js';

const LOOP_S = 20;

// ---------- boot ----------
const robot = await buildRobot('assets/robot.json', 'assets/meshes/');
robot.root.rotation.set(-Math.PI / 2, 0, 0, 'YXZ'); // MJCF z-up → three y-up

const gait = new GaitController(robot);
const cns = new Connectome();
window.__dbg = { gait, robot, cns, THREE };
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
simCam.position.set(0.32, 0.19, 0.44);
const orbit = new OrbitControls(simCam, simCanvas);
orbit.target.set(0, 0.09, 0);
orbit.enableDamping = true;
orbit.minDistance = 0.15;
orbit.maxDistance = 3;
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

simScene.add(new THREE.AmbientLight(0xbfd4e8, 0.32));
const key = new THREE.SpotLight(0xffffff, 26, 0, 0.42, 0.85, 1.4);
key.position.set(0.9, 2.2, 0.7);
key.castShadow = true;
key.shadow.mapSize.set(1024, 1024);
key.target.position.set(0, 0, 0);
simScene.add(key, key.target);
const fill = new THREE.DirectionalLight(0x88aacc, 0.5);
fill.position.set(-1, 0.8, -0.6);
simScene.add(fill);
const rim = new THREE.DirectionalLight(0x56d9d3, 0.6);
rim.position.set(-0.4, 0.5, 1);
simScene.add(rim);

simScene.add(robot.root);

// ---------- CNS scenes ----------
const cnsScene = new THREE.Scene();
cnsScene.background = new THREE.Color('#070b12');
cnsScene.add(cns.group);

const cnsFullCanvas = document.getElementById('cnsFullCanvas');
const cnsVncCanvas = document.getElementById('cnsVncCanvas');
const cnsFullCam = new THREE.PerspectiveCamera(30, 1, 10, 8000);
cnsFullCam.position.set(0, -60, 2720);
cnsFullCam.lookAt(20, -70, 0);
const cnsVncCam = new THREE.PerspectiveCamera(30, 1, 10, 8000);
cnsVncCam.position.set(-880, -330, 1950);
cnsVncCam.lookAt(-20, -240, 0);

// ---------- renderers ----------
function makeRenderer(canvas) {
  const r = new THREE.WebGLRenderer({ canvas, antialias: true });
  r.setPixelRatio(Math.min(devicePixelRatio, 2));
  r.shadowMap.enabled = canvas === simCanvas;
  r.shadowMap.type = THREE.PCFSoftShadowMap;
  return r;
}
const simR = makeRenderer(simCanvas);
const fullR = makeRenderer(cnsFullCanvas);
const vncR = makeRenderer(cnsVncCanvas);

function fitRenderer(r, canvas, cam) {
  const box = canvas.parentElement.getBoundingClientRect();
  const w = Math.max(2, box.width), h = Math.max(2, box.height);
  r.setSize(w, h, false);
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
const state = {
  playing: true,
  chips: { wireframe: false, connectome: false, controller: true, physics: false, ocean: true },
};

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
      gait.trained = on;
      break;
    case 'physics': gait.physics = on; break;
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

function frame() {
  requestAnimationFrame(frame);
  const dt = Math.min(clock.getDelta(), 0.05);

  let snap = null;
  if (state.playing) {
    snap = gait.update(dt);
    simTime = (simTime + dt) % LOOP_S;
    timeline.record(snap.angles, snap.drives);
    const hr = (snap.body.heading - prevHeading) / Math.max(dt, 1e-4);
    headingRate += (hr - headingRate) * Math.min(1, dt * 4);
    prevHeading = snap.body.heading;
  } else if (scrubFrac !== null) {
    timeline.sampleAt(scrubFrac, replayAngles, replayDrives);
    for (const j of JOINT_ORDER) robot.setJoint(j, replayAngles[j]);
    snap = gait.snapshot();
    snap.drives = { ...replayDrives };
  } else {
    snap = gait.snapshot();
  }

  cns.update({
    drives: snap.drives,
    signals: {
      turn: THREE.MathUtils.clamp(headingRate * 0.8, -1, 1),
      scan: snap.scanYaw / 0.9,
      speed: snap.body.speed / 0.06,
    },
  }, dt);

  // HUD
  const epT = simTime;
  hudTime.textContent = epT.toFixed(2).padStart(5, '0') + 's';
  hudDuration.textContent = snap.eventDuration.toFixed(2) + 's';
  hudBody.textContent =
    `${Math.round(((snap.body.heading * 180 / Math.PI) % 360 + 360) % 360 * 100)}/${Math.round(snap.body.pitch * 180 / Math.PI * 100)}`;
  // most active descending unit = current "model unit"
  let best = 0, bestA = -1;
  for (let i = 56; i < 72; i++) {
    const a = Math.abs(cns.activity[i]);
    if (a > bestA) { bestA = a; best = i; }
  }
  hudUnit.textContent = best;
  hudState.textContent = bestA.toFixed(3);
  tcNow.textContent = fmtTime(epT);
  scrubPlayed.style.width = `${(epT / LOOP_S) * 100}%`;

  if (state.playing) timeline.draw();

  followRobot();
  orbit.update();
  simR.render(simScene, simCam);
  fullR.render(cnsScene, cnsFullCam);
  vncR.render(cnsScene, cnsVncCam);
}
frame();
