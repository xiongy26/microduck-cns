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

// Static box obstacles scattered beyond the maze exit, in MuJoCo z-up world
// coords (half-sizes). physics.js injects them as collision geoms; main.js
// builds the matching visual meshes from the same array, so rays that see a
// rock are rays that hit a rock.
export const OBSTACLES = [
  { pos: [1.8, 5.6, 0.15], size: [0.17, 0.21, 0.15] },
  { pos: [2.8, 6.8, 0.17], size: [0.23, 0.16, 0.17] },
  { pos: [1.0, 7.4, 0.16], size: [0.15, 0.15, 0.16] },
  { pos: [2.4, 8.2, 0.16], size: [0.19, 0.23, 0.16] },
  { pos: [3.4, 6.2, 0.16], size: [0.21, 0.17, 0.16] },
];

// The maze: an L-shaped corridor with one dead-end pocket on the south wall.
// Spawn is the STAND keyframe (0,0) facing +x, deep in the long arm; the only
// exit opens north at the top of the short arm (y ≈ 4.4). Corridors are
// 2.4 m wide so centered walls sit just beyond the 1.3 m whisker range —
// the brain is blind while centered, senses a wall only when drifting toward
// it, and its avoid-steer re-centers the body. Wall tops are 0.45 m, well
// above the ray height (~0.25 m). 2-D closed-loop tuning lives in
// tools/maze_sim.mjs (escape ≈ 70% within 3 min over random seeds).
export const MAZE_WALLS = [
  { pos: [-0.05, 1.3, 0.225], size: [0.75, 0.1, 0.225] },  // A2 long-arm north
  { pos: [0.7, 2.3, 0.225],   size: [0.1, 1.1, 0.225] },   // B1 short-arm west
  { pos: [3.3, 2.8, 0.225],   size: [0.1, 1.6, 0.225] },   // B2 short-arm east
  { pos: [3.3, -0.1, 0.225],  size: [0.1, 1.3, 0.225] },   // A3 long-arm east cap
  { pos: [0.2, -1.3, 0.225],  size: [0.8, 0.1, 0.225] },   // A1a long-arm south w/ pocket mouth
  { pos: [2.7, -1.3, 0.225],  size: [0.5, 0.1, 0.225] },   // A1b south-east
  { pos: [0.9, -1.9, 0.225],  size: [0.1, 0.7, 0.225] },   // N1 pocket west
  { pos: [2.3, -1.9, 0.225],  size: [0.1, 0.7, 0.225] },   // N2 pocket east
  { pos: [1.6, -2.7, 0.225],  size: [0.8, 0.1, 0.225] },   // N3 pocket end
  { pos: [-0.7, 0.0, 0.225],  size: [0.1, 1.4, 0.225] },   // A4 west cap (behind spawn)
];

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
  // static collision boxes for the maze walls and the rocks beyond the exit
  for (const [name, arr] of [['wall', MAZE_WALLS], ['rock', OBSTACLES]]) {
    arr.forEach(({ pos, size }, i) => {
      world.appendChild(el('geom', {
        name: `${name}${i}`, type: 'box',
        size: size.join(' '), pos: pos.join(' '),
        rgba: name === 'wall' ? '0.2 0.25 0.33 1' : '0.16 0.2 0.26 1',
      }));
    });
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
  // velocity commands come from the connectome brain (main.js每帧写入):
  // the brain decides walk/stand/turn, this policy handles low-level balance.
  const command = { vx: 0, vy: 0, wz: 0 };
  let headOffsets = null;  // {neck_pitch, head_pitch, head_yaw, head_roll} from the brain
  let stepCount = 0;
  let paused = false;
  let mode = 'stand';      // derived from the brain's commands (walk/stand)
  let modeSince = 0;       // control steps since last mode switch
  let running = false;
  // fall detection + self-recovery (1:1 with the reference pipeline):
  // recovery = null → normal; {state:"fallen"} → let physics settle;
  // {state:"recovering"} → the stand policy self-rights the duck.
  let recovery = null;
  let fallDebounce = 0;
  let prevX = 0, prevY = 0;
  const FALL_DEBOUNCE_STEPS = 10;
  const FALL_SETTLE_STEPS = 15;
  const RECOVER_UPRIGHT_STEPS = 50;
  const RECOVER_GIVEUP_STEPS = 300;
  const _q = new THREE.Quaternion();
  const _g = new THREE.Vector3();
  const HEAD_JOINTS = ['neck_pitch', 'head_pitch', 'head_yaw', 'head_roll'];

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
    modeSince = 0;
    prevX = data.qpos[0]; prevY = data.qpos[1];
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
    for (let j = 0; j < 14; j++) {
      // brain head offsets ride on top of the policy's own head actions
      const off = headOffsets?.[JOINT_NAMES[j]] ?? 0;
      data.ctrl[j] = DEFAULT_POSE[j] + act[j] + off;
    }
    for (let s = 0; s < DECIMATION; s++) mujoco.mj_step(model, data);
    stepCount++;
    // the brain's velocity commands define the behavioral mode
    const newMode = (Math.abs(command.vx) > 0.03 || Math.abs(command.wz) > 0.02) ? 'walk' : 'stand';
    if (newMode !== mode) { mode = newMode; modeSince = 0; } else modeSince++;
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
    // velocity commands from the connectome brain (called every animation frame)
    setCommand(vx, vy, wz) {
      command.vx = vx; command.vy = vy; command.wz = wz;
    },
    // head joint target offsets from the brain's gaze-scan circuit
    setHeadOffsets(head) { headOffsets = head; },
    get policy() { return recovery?.state === 'recovering' ? 'stand (recovery)' : 'walk'; },
    get mode() { return recovery ? 'recovery' : mode; },
    get modeSinceSteps() { return modeSince; },
    get fallen() { return !!recovery; },
    // current state snapshot for the renderer
    getState() {
      const qpos = data.qpos;
      const angles = new Float32Array(14);
      for (let j = 0; j < 14; j++) angles[j] = qpos[qposAdr[j]];
      // sensory channels for the brain: base speed (horizontal), body yaw rate
      // (imu gyro z, body frame ≈ world frame near upright), upright measure
      const gz = projGravZ();
      const sens = data.sensordata;
      const speed = Math.hypot(qpos[0] - prevX, qpos[1] - prevY) / CTRL_DT;
      prevX = qpos[0]; prevY = qpos[1];
      let headSpeed = 0;
      for (let j = 5; j <= 8; j++) headSpeed += Math.abs(data.qvel[dofAdr[j]]);
      headSpeed /= 4;
      return {
        pos: [qpos[0], qpos[1], qpos[2]],
        quat: [qpos[3], qpos[4], qpos[5], qpos[6]], // wxyz, MuJoCo world frame
        angles,
        action: Float32Array.from(lastAction),
        time: stepCount * CTRL_DT,
        mode,
        upright: data.qpos[2],
        speed,
        yawRate: sens[gyroAdr + 2],
        gravityZ: gz,
        headSpeed,
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
