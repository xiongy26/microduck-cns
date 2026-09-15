// physics.js — real MuJoCo physics + trained RL policies for the MicroDuck.
// Ported 1:1 from the proven pipeline in ~/humanoid-robot/microduck-ar
// (src/sim.js duck robot config): MuJoCo WASM, 61-D observations
// [base_ang_vel(3), projected_gravity(3), joint_pos(14), joint_vel(14),
//  last_action(14), command(13)], ONNX actors via onnxruntime-web, 50 Hz
// policy with 4×5 ms physics substeps, position actuators from the MJCF,
// STAND keyframe reset. The duck policies (BEST_alpha_stand /
// BEST_alpha_walking) are the trained "alpha" actors.
import * as THREE from 'three';

export const JOINT_NAMES = [
  'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
  'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
  'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle',
];
export const DEFAULT_POSE = new Float32Array([
  0, -0.08726646259971647, -0.457924, -0.004940, 0.452984,
  0.3490658503988659, 0.3490658503988659, 0, 0,
  0, 0.08726646259971647, 0.457924, 0.004940, -0.452984,
]);
const CTRL_DT = 0.02;   // 50 Hz policy
const DECIMATION = 4;   // ×5 ms physics substeps
const OBS_SIZE = 61;
const MUJOCO_URL = new URL('../assets/vendor/mujoco/mujoco.js', import.meta.url).href;
const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.min.mjs';
const ORT_WASM_DIR = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';
const POLICY_FILES = {
  stand: 'assets/policies/BEST_alpha_stand.onnx',
  walk: 'assets/policies/BEST_alpha_walking.onnx',
};

export async function createDuckPhysics({ onProgress = () => {} } = {}) {
  const step = (msg) => onProgress(msg);

  // ── runtimes ────────────────────────────────────────────────────────────
  step('loading MuJoCo WASM');
  const mujocoModule = await import(MUJOCO_URL);
  const mujoco = await mujocoModule.default();
  step('loading onnxruntime-web');
  const ort = await import(/* @vite-ignore */ ORT_URL);
  ort.env.wasm.wasmPaths = ORT_WASM_DIR;
  ort.env.wasm.numThreads = 1; // static hosting sends no COOP/COEP headers

  // ── physics MJCF: strip visual geoms, add floor + STAND keyframe ───────
  step('preparing physics model');
  const src = await (await fetch('assets/mjcf/robot_allcollisions.xml')).text();
  const doc = new DOMParser().parseFromString(src, 'text/xml');
  for (const g of [...doc.querySelectorAll('geom[class="visual"]')]) g.remove();
  const usedMeshes = new Set(
    [...doc.querySelectorAll('geom[mesh]')].map((g) => g.getAttribute('mesh')),
  );
  for (const m of [...doc.querySelectorAll('asset > mesh')]) {
    const name = m.getAttribute('name') ?? m.getAttribute('file').replace(/\.stl$/i, '');
    if (!usedMeshes.has(name)) m.remove();
  }
  const el = (tag, attrs) => {
    const e = doc.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  };
  doc.documentElement.appendChild(el('option', { timestep: '0.005' }));
  const world = doc.querySelector('worldbody');
  if (!doc.querySelector('geom[name="floor"]')) {
    world.appendChild(el('geom', { name: 'floor', type: 'plane', size: '0 0 0.05', pos: '0 0 0' }));
  }
  const poseByName = new Map(JOINT_NAMES.map((n, i) => [n, DEFAULT_POSE[i]]));
  const qposJoints = [...doc.querySelectorAll('body > joint')]
    .filter((j) => j.getAttribute('type') !== 'free')
    .map((j) => poseByName.get(j.getAttribute('name')) ?? 0)
    .join(' ');
  const keyframe = el('key', {
    name: 'STAND',
    qpos: `0 0 0.12 1 0 0 0 ${qposJoints}`.trim(),
    ctrl: Array.from(DEFAULT_POSE).join(' '),
  });
  const keyframes = doc.createElement('keyframe');
  keyframes.appendChild(keyframe);
  doc.documentElement.appendChild(keyframes);
  const meshFiles = [...doc.querySelectorAll('asset > mesh')].map((m) => m.getAttribute('file'));
  const xml = new XMLSerializer().serializeToString(doc);

  // ── collision meshes into the MuJoCo VFS ────────────────────────────────
  step('building collision meshes');
  const vfs = new mujoco.MjVFS();
  await Promise.all(meshFiles.map(async (f) => {
    const buf = await (await fetch(`assets/mjcf/meshes/${f}`)).arrayBuffer();
    vfs.addBuffer(`assets/${f}`, new Uint8Array(buf));
  }));

  step('compiling physics');
  const model = mujoco.MjModel.from_xml_string(xml, vfs);
  const data = new mujoco.MjData(model);

  const qposAdr = JOINT_NAMES.map((n) => model.jnt(n).qposadr);
  const dofAdr = JOINT_NAMES.map((n) => model.jnt(n).dofadr);
  const gyroAdr = model.sensor('imu_ang_vel').adr;
  const trunkId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'trunk_base');
  const standKeyId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY.value, 'STAND');

  // ── policy sessions ─────────────────────────────────────────────────────
  step('loading RL policies');
  const sessions = {};
  for (const [name, file] of Object.entries(POLICY_FILES)) {
    sessions[name] = await ort.InferenceSession.create(file, { executionProviders: ['wasm'] });
  }

  // ── state ───────────────────────────────────────────────────────────────
  let lastAction = new Float32Array(14);
  const obs = new Float32Array(OBS_SIZE);
  const cmd = new Float32Array(13);
  const command = { vx: 0, vy: 0, wz: 0 };
  let commandOverride = false;
  let stepCount = 0;
  let paused = false;
  let mode = 'stand';      // exploratory behavior cycle: stand <-> walk
  let modeTimer = 2.5;
  let modeSince = 0;       // control steps since last mode switch
  let running = false;
  // fall detection + self-recovery (1:1 with the reference pipeline):
  // recovery = null → normal; {state:"fallen"} → let physics settle;
  // {state:"recovering"} → the stand policy self-rights the duck.
  let recovery = null;
  let fallDebounce = 0;
  let turnWz = 0;
  const FALL_DEBOUNCE_STEPS = 10;
  const FALL_SETTLE_STEPS = 15;
  const RECOVER_UPRIGHT_STEPS = 50;
  const RECOVER_GIVEUP_STEPS = 300;
  const _q = new THREE.Quaternion();
  const _g = new THREE.Vector3();

  function projGravZ() {
    const xq = data.body(trunkId).xquat;
    _q.set(xq[1], xq[2], xq[3], xq[0]).conjugate();
    _g.set(0, 0, -1).applyQuaternion(_q);
    return _g.z;
  }

  function reset() {
    mujoco.mj_resetDataKeyframe(model, data, standKeyId);
    mujoco.mj_forward(model, data);
    lastAction.fill(0);
    recovery = null;
    fallDebounce = 0;
    mode = 'stand';
    modeTimer = 2;
    modeSince = 0;
  }

  function buildObs() {
    const qpos = data.qpos, qvel = data.qvel, sens = data.sensordata;
    let i = 0;
    for (let a = 0; a < 3; a++) obs[i++] = sens[gyroAdr + a];
    const xq = data.body(trunkId).xquat; // [w, x, y, z]
    _q.set(xq[1], xq[2], xq[3], xq[0]).conjugate();
    _g.set(0, 0, -1).applyQuaternion(_q);
    obs[i++] = _g.x; obs[i++] = _g.y; obs[i++] = _g.z;
    for (let j = 0; j < 14; j++) obs[i++] = qpos[qposAdr[j]] - DEFAULT_POSE[j];
    for (let j = 0; j < 14; j++) obs[i++] = qvel[dofAdr[j]];
    for (let j = 0; j < 14; j++) obs[i++] = lastAction[j];
    cmd.fill(0);
    // zero command while recovering (the stand policy needs it to self-right)
    if (!recovery) {
      cmd[0] = command.vx; cmd[1] = command.vy; cmd[2] = command.wz;
    }
    for (let c = 0; c < 13; c++) obs[i++] = cmd[c];
    return obs;
  }

  function poseIsDead() {
    const z = data.qpos[2];
    const gz = projGravZ();
    if (!Number.isFinite(z) || !Number.isFinite(gz)) return 'exploded';
    if (gz > -0.5 || z < 0.02) return 'fallen';
    return null;
  }

  async function controlStep() {
    // policy selection: walk actor for the behavior cycle, stand actor while
    // self-righting after a fall (1:1 with the reference activeSession())
    const session = recovery?.state === 'recovering' ? sessions.stand : sessions.walk;
    const feeds = { obs: new ort.Tensor('float32', buildObs(), [1, OBS_SIZE]) };
    const out = await session.run(feeds);
    const act = Object.values(out)[0].data;
    lastAction.set(act);
    for (let j = 0; j < 14; j++) data.ctrl[j] = DEFAULT_POSE[j] + act[j];
    for (let s = 0; s < DECIMATION; s++) mujoco.mj_step(model, data);
    stepCount++;
    modeSince++;
    // exploratory behavior: stand in place <-> walk straight, occasionally
    // turning. Command magnitudes are within the proven keyboard range of the
    // reference pipeline (vx 0.25; small wz bursts) — the walk policy ignores
    // weaker commands.
    if (!recovery && !commandOverride) {
      modeTimer -= CTRL_DT;
      if (modeTimer <= 0) {
        if (mode === 'stand') {
          mode = 'walk';
          modeTimer = 6 + Math.random() * 4;
          turnWz = Math.random() < 0.35 ? (Math.random() < 0.5 ? 0.35 : -0.35) : 0;
        } else {
          mode = 'stand';
          modeTimer = 2.5 + Math.random() * 3;
        }
        modeSince = 0;
      }
      if (mode === 'walk') {
        command.vx = 0.25;
        command.vy = 0;
        command.wz = turnWz;
      } else {
        command.vx = 0; command.vy = 0; command.wz = 0;
      }
    }
    // fall handling
    const death = poseIsDead();
    if (death === 'exploded') { reset(); return; }
    if (recovery) {
      recovery.steps++;
      if (recovery.state === 'fallen') {
        if (recovery.steps >= FALL_SETTLE_STEPS) {
          recovery = { state: 'recovering', steps: 0, uprightSteps: 0 };
          lastAction.fill(0);
        }
      } else {
        recovery.uprightSteps = projGravZ() < -0.85 ? recovery.uprightSteps + 1 : 0;
        if (recovery.uprightSteps >= RECOVER_UPRIGHT_STEPS) {
          recovery = null;
          mode = 'walk';
          modeTimer = 5;
          modeSince = 0;
          lastAction.fill(0);
        } else if (recovery.steps >= RECOVER_GIVEUP_STEPS) {
          reset();
        }
      }
    } else if (death === 'fallen') {
      if (++fallDebounce >= FALL_DEBOUNCE_STEPS) {
        fallDebounce = 0;
        recovery = { state: 'fallen', steps: 0 };
        mode = 'stand';
      }
    } else {
      fallDebounce = 0;
    }
  }

  async function loop() {
    while (running) {
      if (paused) { await new Promise((r) => setTimeout(r, 50)); continue; }
      const t0 = performance.now();
      try {
        await controlStep();
      } catch (err) {
        console.error('[physics] control step failed', err);
        await new Promise((r) => setTimeout(r, 200));
        continue;
      }
      const wait = CTRL_DT * 1000 - (performance.now() - t0);
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    }
  }

  reset();

  return {
    JOINT_NAMES,
    mujoco, model, data,
    start() { if (!running) { running = true; loop(); } },
    pause() { paused = true; },
    resume() { paused = false; },
    reset,
    setCommand(vx, vy, wz) {
      commandOverride = true;
      command.vx = vx; command.vy = vy; command.wz = wz;
    },
    clearCommandOverride() { commandOverride = false; },
    forceMode(m) { mode = m; modeTimer = 30; modeSince = 0; }, // debug override
    get policy() { return recovery?.state === 'recovering' ? 'stand (recovery)' : 'walk'; },
    get mode() { return recovery ? 'recovery' : mode; },
    get modeSinceSteps() { return modeSince; },
    get fallen() { return !!recovery; },
    // current state snapshot for the renderer
    getState() {
      const qpos = data.qpos;
      const angles = new Float32Array(14);
      for (let j = 0; j < 14; j++) angles[j] = qpos[qposAdr[j]];
      return {
        pos: [qpos[0], qpos[1], qpos[2]],
        quat: [qpos[3], qpos[4], qpos[5], qpos[6]], // wxyz, MuJoCo world frame
        angles,
        action: Float32Array.from(lastAction),
        time: stepCount * CTRL_DT,
        mode,
        upright: data.qpos[2],
      };
    },
  };
}

// MuJoCo world pose -> three.js world pose for the robot wrapper.
// The wrapper does double duty: it carries the MJCF z-up -> three y-up frame
// conversion C = rotX(-90°) AND the body's world rotation qW, so its rotation
// is qC·qW (the child subtree stores raw MJCF coordinates). The trunk_base
// group keeps its baked local translation (0, 0, 0.12 m) in wrapper space,
// which the wrapper position must subtract in world coords.
const _qC = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
const _qW = new THREE.Quaternion();
const _qWc = new THREE.Quaternion();
const _off = new THREE.Vector3();
export function mjcfToWorld(pos, quatWxyz, outPos, outQuat) {
  _qW.set(quatWxyz[1], quatWxyz[2], quatWxyz[3], quatWxyz[0]);
  _qWc.copy(_qC).multiply(_qW);
  outQuat.copy(_qWc);
  _off.set(0, 0, 0.12).applyQuaternion(_qWc);
  outPos.set(pos[0], pos[2], -pos[1]).sub(_off);
}
