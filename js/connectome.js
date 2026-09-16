// connectome.js — a 192-neuron Male CNS subset rendered as anatomical skeletons
// AND simulated as the robot's brain. Layout follows real male fly anatomy:
// brain (supraesophageal ganglion + SOG), cervical connective, and a
// thoracic+abdominal ventral nerve cord with lateral nerve roots.
//
// The same 192 cells / 2,456 weighted connections that are drawn are the
// controller: a continuous-time recurrent network (CTRNN) with functional
// microcircuits mapped onto the populations —
//   brain      0-55   mode circuit (walk W / stand S), steering (L/R),
//                     gaze scan oscillators (SY/SP), association (A)
//   descending 56-71  command pathway W/S/L/R → decoded into vx, wz
//   motor      72-167 one pool per MicroDuck joint, activity = muscle state
//   local      168-175 VNC interneurons
//   ascending  176-191 proprioceptive channels (speed, turn, upright, fall,
//                     head) feeding the brain — the sensory feedback loop
// Slow "homeostat" variables (walk/stand/turn pressure) gate the
// winner-take-all circuits and produce exploratory bouts. Commands are read
// out of the DESCENDING population and sent to the RL policy (low-level
// balance/gait) as velocity commands + head joint offsets.
//
// Swap-in point for real data: replace growNeuron() skeletons and the edge
// list with MANC skeletons/connections from neuPrint (neuprint-cns.janelia.org,
// dataset "manc"); a token is required, see README.
import * as THREE from 'three';

export const NEURON_COUNT = 192;
export const CONNECTION_COUNT = 2456;

// populations: [start, end) index ranges
export const POPS = {
  brain: [0, 56],
  descending: [56, 72],
  motor: [72, 168],
  local: [168, 176],
  ascending: [176, 192],
};

// functional microcircuit groups (indices within brain/descending)
// W walk drive, S stand/rest, L/R steer left/right, SY/SP1/SP2 gaze scan
// (yaw/pitch/roll), A association; DW/DS/DL/DR descending command cells
const G = {
  W: [0, 8], S: [8, 16], L: [16, 24], R: [24, 32],
  SY: [32, 40], SP1: [40, 44], SP2: [44, 48], A: [48, 56],
  DW: [56, 60], DS: [60, 64], DL: [64, 68], DR: [68, 72],
};
// ascending sensory channels
const ASC = { fwd: [176, 180], turnL: [180, 182], turnR: [182, 184], up: [184, 186], fall: [186, 188], head: [188, 192] };

// 14 joint pools mapped onto the cord (foreleg → pro/meso/metathoracic, head → SOG)
const POOL_LAYOUT = [
  { j: 'left_hip_yaw', n: 7, seg: 'pro' }, { j: 'left_hip_roll', n: 7, seg: 'pro' },
  { j: 'left_hip_pitch', n: 7, seg: 'meso' }, { j: 'left_knee', n: 7, seg: 'meso' },
  { j: 'left_ankle', n: 7, seg: 'meta' },
  { j: 'right_hip_yaw', n: 7, seg: 'pro' }, { j: 'right_hip_roll', n: 7, seg: 'pro' },
  { j: 'right_hip_pitch', n: 7, seg: 'meso' }, { j: 'right_knee', n: 7, seg: 'meso' },
  { j: 'right_ankle', n: 7, seg: 'meta' },
  { j: 'neck_pitch', n: 6, seg: 'sog' }, { j: 'head_pitch', n: 6, seg: 'sog' },
  { j: 'head_yaw', n: 6, seg: 'sog' }, { j: 'head_roll', n: 6, seg: 'sog' },
];
// 10*7 + 4*6 = 94; POPS.motor has 96 slots — last 2 become local interneurons

// ---- circuit parameters (tuned in tools/ctrnn prototype; see README) ----
// homeostats [accumulate rate, decay rate, gain, init] — slow bout pressure
// decay rates bumped up so the W/S walk-vs-stand bouts become visible within a
// few seconds instead of half a minute; init values halved so the very first
// frames already show meaningful activity in the brain populations.
const HOMEOSTAT = {
  hw: [0.18, 0.18, 2.4, 0.6],  // walk pressure: inhibits W while walking
  hs: [0.18, 0.30, 2.4, 0.30], // stand pressure: inhibits S while standing
  hl: [0.40, 0.40, 1.8, 0.0],  // turn-left pressure
  hr: [0.40, 0.40, 1.8, 0.0],  // turn-right pressure
};
const SENSORY_GAIN = 3.0;

const CYAN = new THREE.Color('#3fd8d0');
const ORANGE = new THREE.Color('#ff8a3c');
// brighter dim colors so even quiet / resting neurons are clearly readable on
// the dark background — the old dim teal/brown was nearly invisible
const CYAN_DIM = new THREE.Color('#5ec8be');
const ORANGE_DIM = new THREE.Color('#d68a5a');

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Connectome {
  constructor(seed = 19200301) {
    this.rng = mulberry32(seed);
    this.neurons = [];
    this.growAll();
    this.edges = this.wire();          // [src, dst, weight] — the drawn AND simulated network
    this.buildState();
    this.activity = new Float32Array(NEURON_COUNT);
    this.cmds = {
      vx: 0, wz: 0, walkDrive: 0, mode: 'stand',
      head: { neck_pitch: 0, head_pitch: 0, head_yaw: 0, head_roll: 0 },
    };
    this.buildScene();
  }

  // ---------- anatomy (procedural skeletons, realData swap-in point) ----------

  growAll() {
    const rng = this.rng;
    const [b0, b1] = POPS.brain;
    for (let i = b0; i < b1; i++) this.neurons.push(this.growBrainNeuron());
    const [d0, d1] = POPS.descending;
    for (let i = d0; i < d1; i++) this.neurons.push(this.growDescendingNeuron());
    // motor pools, topographic along the cord
    let mi = POPS.motor[0];
    this.poolOfCell = new Array(NEURON_COUNT).fill(null);
    for (const pool of POOL_LAYOUT) {
      const yCenter = { pro: 10, meso: -105, meta: -215, sog: 120 }[pool.seg];
      for (let k = 0; k < pool.n; k++) {
        this.neurons.push(this.growMotorNeuron(yCenter + (rng() - 0.5) * 60, mi, pool.j));
        this.poolOfCell[mi] = pool.j;
        mi++;
      }
    }
    while (mi < POPS.motor[1]) { this.neurons.push(this.growMotorNeuron(-150 + (rng() - 0.5) * 120, mi, null)); mi++; }
    const [l0, l1] = POPS.local;
    for (let i = l0; i < l1; i++) this.neurons.push(this.growLocalNeuron());
    const [a0, a1] = POPS.ascending;
    for (let i = a0; i < a1; i++) this.neurons.push(this.growAscendingNeuron());

    // per-neuron render metadata
    for (const n of this.neurons) {
      n.color = new THREE.Color();
      n.wob = this.rng() * Math.PI * 2;    // wobble phase
      n.wobF = 0.5 + this.rng() * 1.5;     // wobble freq
    }
  }

  brainPoint(out) {
    const r = this.rng;
    // two hemispheres + slight A-P elongation; side-view silhouette like FAFB
    const hemi = r() < 0.5 ? -1 : 1;
    const u = r() * 2 - 1, v = r() * 2 - 1, w = r() * 2 - 1;
    out.set(
      60 + u * 190,
      350 + v * 125 + 18 * u,
      hemi * (28 + Math.abs(w) * 72)
    );
    return out;
  }

  growBrainNeuron() {
    const r = this.rng;
    const soma = this.brainPoint(new THREE.Vector3());
    const segs = [];
    const nBranch = 5 + Math.floor(r() * 8);
    for (let b = 0; b < nBranch; b++) {
      let p = soma.clone();
      let dir = new THREE.Vector3(r() * 2 - 1, r() * 2 - 1, r() * 2 - 1).normalize();
      const nSeg = 3 + Math.floor(r() * 4);
      for (let s = 0; s < nSeg; s++) {
        const q = p.clone().addScaledVector(dir, 16 + r() * 30);
        // stay inside the brain volume
        const dx = (q.x - 60) / 210, dy = (q.y - 350) / 150, dz = q.z / 110;
        if (dx * dx + dy * dy + dz * dz > 1) break;
        segs.push(p.clone(), q.clone());
        p = q;
        dir.add(new THREE.Vector3(r() * 2 - 1, r() * 2 - 1, r() * 2 - 1).multiplyScalar(0.7)).normalize();
      }
    }
    if (segs.length < 2) segs.push(soma.clone(), soma.clone().add(new THREE.Vector3(30, 10, 0)));
    return { soma, segs, pop: 'brain', pool: null };
  }

  growDescendingNeuron() {
    const r = this.rng;
    const soma = new THREE.Vector3(30 + (r() - 0.5) * 120, 300 + r() * 60, (r() - 0.5) * 90);
    const segs = [];
    let p = soma.clone();
    const yEnd = 60 + r() * 80;
    const xEnd = 20 + (r() - 0.5) * 40;
    while (p.y > yEnd) {
      const q = p.clone().add(new THREE.Vector3((r() - 0.5) * 26, -(28 + r() * 30), (r() - 0.5) * 20));
      segs.push(p.clone(), q.clone());
      p = q;
    }
    // terminal arbor
    for (let b = 0; b < 3; b++) {
      const q = p.clone().add(new THREE.Vector3((r() - 0.5) * 70, -r() * 40, (r() - 0.5) * 50));
      segs.push(p.clone(), q.clone());
    }
    return { soma, segs, pop: 'descending', pool: null };
  }

  growMotorNeuron(yCenter, idx, pool) {
    const r = this.rng;
    const side = idx % 2 === 0 ? 1 : -1;
    const soma = new THREE.Vector3(
      (r() - 0.5) * 120,
      yCenter + (r() - 0.5) * 40,
      side * (20 + r() * 45)
    );
    const segs = [];
    // dendritic arbor within the neuromere
    const nBranch = 4 + Math.floor(r() * 6);
    for (let b = 0; b < nBranch; b++) {
      let p = soma.clone();
      let dir = new THREE.Vector3(r() * 2 - 1, r() * 2 - 1, r() * 2 - 1).normalize();
      const nSeg = 2 + Math.floor(r() * 4);
      for (let s = 0; s < nSeg; s++) {
        const q = p.clone().addScaledVector(dir, 14 + r() * 26);
        segs.push(p.clone(), q.clone());
        p = q;
        dir.add(new THREE.Vector3(r() * 2 - 1, r() * 2 - 1, r() * 2 - 1).multiplyScalar(0.8)).normalize();
      }
    }
    // lateral nerve root: axon exits the cord and branches (peripheral nerve)
    let p = soma.clone();
    const outZ = side * (95 + r() * 60);
    while (Math.abs(p.z) < Math.abs(outZ)) {
      const q = p.clone().add(new THREE.Vector3((r() - 0.5) * 18, (r() - 0.5) * 22, side * (18 + r() * 20)));
      segs.push(p.clone(), q.clone());
      p = q;
    }
    for (let b = 0; b < 2 + Math.floor(r() * 3); b++) {
      const q = p.clone().add(new THREE.Vector3((r() - 0.5) * 60, (r() - 0.5) * 50, side * (10 + r() * 40)));
      segs.push(p.clone(), q.clone());
    }
    return { soma, segs, pop: 'motor', pool };
  }

  growLocalNeuron() {
    const r = this.rng;
    const soma = new THREE.Vector3((r() - 0.5) * 100, -140 + (r() - 0.5) * 220, (r() - 0.5) * 70);
    const segs = [];
    for (let b = 0; b < 5; b++) {
      let p = soma.clone();
      for (let s = 0; s < 3; s++) {
        const q = p.clone().add(new THREE.Vector3((r() - 0.5) * 50, (r() - 0.5) * 40, (r() - 0.5) * 40));
        segs.push(p.clone(), q.clone());
        p = q;
      }
    }
    return { soma, segs, pop: 'local', pool: null };
  }

  growAscendingNeuron() {
    const r = this.rng;
    const soma = new THREE.Vector3((r() - 0.5) * 100, -40 + (r() - 0.5) * 140, (r() - 0.5) * 60);
    const segs = [];
    let p = soma.clone();
    while (p.y < 240) {
      const q = p.clone().add(new THREE.Vector3((r() - 0.5) * 24, 26 + r() * 30, (r() - 0.5) * 18));
      segs.push(p.clone(), q.clone());
      p = q;
    }
    for (let b = 0; b < 3; b++) {
      const q = p.clone().add(new THREE.Vector3((r() - 0.5) * 80, r() * 40, (r() - 0.5) * 40));
      segs.push(p.clone(), q.clone());
    }
    return { soma, segs, pop: 'ascending', pool: null };
  }

  // ---------- wiring: exactly CONNECTION_COUNT weighted directed edges ----------
  // Functional microcircuits are realized as structured edge sets inside the
  // same population categories the dashboard counts; the remainder is diffuse
  // weak coupling (the rng here continues the anatomy's stream, so a given
  // seed yields one fixed connectome).

  wire() {
    const rng = this.rng;
    const edges = [];
    const grp = (name) => Array.from({ length: G[name][1] - G[name][0] }, (_, k) => G[name][0] + k);
    const allOf = (pop) => Array.from({ length: POPS[pop][1] - POPS[pop][0] }, (_, k) => POPS[pop][0] + k);
    const ascGrp = (name) => Array.from({ length: ASC[name][1] - ASC[name][0] }, (_, k) => ASC[name][0] + k);
    const pairs = (srcs, dsts, count, w) => {
      for (let k = 0; k < count; k++) {
        edges.push([
          srcs[Math.floor(rng() * srcs.length)],
          dsts[Math.floor(rng() * dsts.length)],
          typeof w === 'function' ? w() : w,
        ]);
      }
    };
    const poolCells = {};
    {
      let mi = POPS.motor[0];
      for (const { j, n } of POOL_LAYOUT) { poolCells[j] = []; for (let k = 0; k < n; k++) poolCells[j].push(mi++); }
    }

    // --- brain→brain (520): WTA mode & steering circuits, scan oscillators ---
    {
      const S = [
        ['W', 'W', 24, +0.20], ['S', 'S', 24, +0.20],
        ['W', 'S', 32, -0.25], ['S', 'W', 32, -0.25],
        ['L', 'R', 32, -0.25], ['R', 'L', 32, -0.25],
        ['L', 'L', 16, +0.10], ['R', 'R', 16, +0.10],
        ['W', 'L', 16, +0.10], ['W', 'R', 16, +0.10],
        ['A', 'W', 12, +0.06], ['A', 'S', 12, +0.06], ['A', 'L', 12, +0.06], ['A', 'R', 12, +0.06],
        ['SY', 'SY', 8, +0.35], ['SP1', 'SP1', 8, +0.35], ['SP2', 'SP2', 8, +0.35],
        ['SY', 'SP1', 4, -0.30], ['SP1', 'SY', 4, -0.30], ['SY', 'SP2', 4, -0.30], ['SP2', 'SY', 4, -0.30],
        ['A', 'A', 8, +0.20],
      ];
      let struct = 0;
      for (const [s, d, c, w] of S) { pairs(grp(s), grp(d), c, w); struct += c; }
      const brainAll = allOf('brain');
      pairs(brainAll, brainAll, 520 - struct, () => (rng() * 2 - 1) * 0.12);
    }
    // --- brain→descending (208): command pathway ---
    {
      const S = [
        ['W', 'DW', 24, +0.5], ['S', 'DS', 24, +0.5],
        ['L', 'DL', 24, +0.5], ['R', 'DR', 24, +0.5],
        ['S', 'DW', 16, -0.6], ['W', 'DL', 12, +0.3], ['W', 'DR', 12, +0.3],
      ];
      let struct = 0;
      for (const [s, d, c, w] of S) { pairs(grp(s), grp(d), c, w); struct += c; }
      pairs(allOf('brain'), allOf('descending'), 208 - struct, () => (rng() * 2 - 1) * 0.2);
    }
    // --- descending→motor (864): command distribution to joint pools ---
    {
      const legs = ['left_hip_pitch', 'left_knee', 'left_ankle', 'right_hip_pitch', 'right_knee', 'right_ankle'].flatMap((j) => poolCells[j]);
      const yawL = poolCells.left_hip_yaw, yawR = poolCells.right_hip_yaw;
      const head = [...poolCells.neck_pitch, ...poolCells.head_pitch, ...poolCells.head_yaw, ...poolCells.head_roll];
      pairs(grp('DW'), legs, 384, +0.15);
      pairs(grp('DL'), yawL, 32, +0.35);
      pairs(grp('DL'), yawR, 32, -0.35);
      pairs(grp('DR'), yawR, 32, +0.35);
      pairs(grp('DR'), yawL, 32, -0.35);
      pairs(grp('DS'), legs, 192, -0.20);
      pairs(grp('DW'), head, 64, +0.15);
      pairs(allOf('descending'), allOf('motor'), 96, () => (rng() * 2 - 1) * 0.15);
    }
    // --- motor→local (96) / local→motor (96): VNC interneuron texture ---
    {
      pairs(allOf('motor'), allOf('local'), 96, () => (rng() * 2 - 1) * 0.3);
      pairs(allOf('local'), allOf('motor'), 96, () => (rng() * 2 - 1) * 0.3);
    }
    // --- motor→ascending (240): proprioceptive feedback channels ---
    {
      const legs = ['left_hip_pitch', 'left_knee', 'left_ankle', 'right_hip_pitch', 'right_knee', 'right_ankle'].flatMap((j) => poolCells[j]);
      const yawPools = [...poolCells.left_hip_yaw, ...poolCells.right_hip_yaw];
      const head = [...poolCells.neck_pitch, ...poolCells.head_pitch, ...poolCells.head_yaw, ...poolCells.head_roll];
      pairs(legs, ascGrp('fwd'), 96, +0.4);
      pairs(yawPools, ascGrp('turnL'), 48, +0.4);
      pairs(yawPools, ascGrp('turnR'), 48, +0.4);
      pairs(head, ascGrp('head'), 48, +0.6);
    }
    // --- ascending→brain (432): sensory feedback into the circuits ---
    {
      const S = [
        ['fwd', 'W', 24, +0.20], ['fwd', 'A', 16, +0.30],
        ['turnL', 'L', 24, -0.45], ['turnR', 'R', 24, -0.45],
        ['up', 'W', 24, +0.22], ['up', 'S', 16, -0.18],
        ['fall', 'S', 24, +1.1], ['fall', 'W', 24, -1.1],
        ['head', 'SY', 12, -0.5], ['head', 'SP1', 8, -0.5], ['head', 'SP2', 8, -0.5],
        ['up', 'A', 16, +0.25],
      ];
      let struct = 0;
      for (const [s, d, c, w] of S) { pairs(ascGrp(s), grp(d), c, w); struct += c; }
      pairs(allOf('ascending'), allOf('brain'), 432 - struct, () => (rng() * 2 - 1) * 0.12);
    }

    if (edges.length !== CONNECTION_COUNT) throw new Error(`connectome: ${edges.length} edges != ${CONNECTION_COUNT}`);
    return edges;
  }

  // ---------- dynamics state ----------

  buildState() {
    const N = NEURON_COUNT;
    this.x = new Float32Array(N);            // membrane/state variables
    this.noise = new Float32Array(N);        // OU process per neuron
    this.t = 0;
    this.hw = HOMEOSTAT.hw[3];               // walk pressure
    this.hs = HOMEOSTAT.hs[3];               // stand pressure
    this.hl = HOMEOSTAT.hl[3];               // steer-left pressure
    this.hr = HOMEOSTAT.hr[3];               // steer-right pressure
    this.walkDrive = 0;                      // previous decoded walk drive (gates scan amplitude)
    this.scanGate = 1;

    // per-neuron params by role
    this.tau = new Float32Array(N).fill(0.3);
    this.bias = new Float32Array(N);
    this.nsig = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      let role = 'A';
      if (i < 8) role = 'W'; else if (i < 16) role = 'S'; else if (i < 24) role = 'L';
      else if (i < 32) role = 'R'; else if (i < 40) role = 'SY'; else if (i < 48) role = 'SP';
      else if (i < 56) role = 'A'; else if (i < 72) role = 'D';
      else if (i < 168) role = 'M'; else if (i < 176) role = 'LOC'; else role = 'ASC';
      const p = {
        W: [1.0, +0.28, 0.08], S: [1.0, +0.22, 0.08],
        L: [0.8, -0.05, 0.35], R: [0.8, -0.05, 0.35],
        SY: [0.25, 0, 0.10], SP: [0.3, 0, 0.10],
        A: [0.6, 0, 0.05], D: [0.2, 0, 0.04],
        M: [0.06, 0, 0.03], LOC: [0.2, 0, 0.05], ASC: [0.08, 0, 0.02],
      }[role];
      this.tau[i] = p[0]; this.bias[i] = p[1]; this.nsig[i] = p[2];
    }

    // gaze-scan oscillators: sinusoid bias per cell (yaw 0.22 Hz, pitch 0.13, roll 0.17)
    this.scan = new Array(N).fill(null);
    for (let k = 0; k < 8; k++) this.scan[G.SY[0] + k] = { f: 0.22, ph: (k / 8) * Math.PI * 2, amp: 0.9 };
    for (let k = 0; k < 4; k++) this.scan[G.SP1[0] + k] = { f: 0.13, ph: (k / 4) * Math.PI * 2, amp: 0.8 };
    for (let k = 0; k < 4; k++) this.scan[G.SP2[0] + k] = { f: 0.17, ph: (k / 4) * Math.PI * 2 + 0.8, amp: 0.8 };

    // incoming edge lists for the recurrent sum
    this.inEdges = Array.from({ length: N }, () => []);
    for (const [s, d, w] of this.edges) this.inEdges[d].push([s, w]);

    this._phi = new Float32Array(N);
    this._drive = new Float32Array(N);
  }

  // ---------- one integration step of the whole CNS ----------
  // sensory: {speed, yawRate, upright, fallen, headSpeed} in [0..~1.2] units
  // drives:  {jointName: [-1..1]} actual muscle states (motor-pool feedback)
  step(sensory, drives, dt) {
    const N = NEURON_COUNT;
    const { x, noise, tau, bias, nsig, scan, inEdges, _phi: phi, _drive: drive } = this;
    const clamp = THREE.MathUtils.clamp;
    this.t += dt;
    const t = this.t;

    // --- external drives ---
    drive.fill(0);
    const sg = clamp(sensory.speed / 0.25, 0, 1.2);
    for (let k = ASC.fwd[0]; k < ASC.fwd[1]; k++) drive[k] += SENSORY_GAIN * sg;
    drive[ASC.turnL[0]] += SENSORY_GAIN * clamp(sensory.yawRate / 0.35, 0, 1.2);
    drive[ASC.turnL[1]] += SENSORY_GAIN * clamp(sensory.yawRate / 0.35, 0, 1.2);
    drive[ASC.turnR[0]] += SENSORY_GAIN * clamp(-sensory.yawRate / 0.35, 0, 1.2);
    drive[ASC.turnR[1]] += SENSORY_GAIN * clamp(-sensory.yawRate / 0.35, 0, 1.2);
    for (let k = ASC.up[0]; k < ASC.up[1]; k++) drive[k] += SENSORY_GAIN * clamp(sensory.upright, 0, 1.2);
    for (let k = ASC.fall[0]; k < ASC.fall[1]; k++) drive[k] += SENSORY_GAIN * clamp(sensory.fallen, 0, 1.2);
    for (let k = ASC.head[0]; k < ASC.head[1]; k++) drive[k] += SENSORY_GAIN * clamp(sensory.headSpeed / 1.5, 0, 1.2);
    // motor pools feel the actual muscle state (their rendered "activity")
    for (let i = POPS.motor[0]; i < POPS.motor[1]; i++) {
      const pool = this.poolOfCell[i];
      drive[i] += 2.2 * (pool && drives[pool] !== undefined ? drives[pool] : 0);
    }

    // --- integrate (semi-implicit Euler, dt already clamped by caller) ---
    for (let i = 0; i < N; i++) phi[i] = Math.tanh(x[i]);
    const scanGate = this.scanGate;
    for (let i = 0; i < N; i++) {
      let I = bias[i] + drive[i];
      if (i < 8) I -= HOMEOSTAT.hw[2] * this.hw;
      else if (i < 16) I -= HOMEOSTAT.hs[2] * this.hs;
      else if (i < 24) I -= HOMEOSTAT.hl[2] * this.hl;
      else if (i < 32) I -= HOMEOSTAT.hr[2] * this.hr;
      const sb = scan[i];
      if (sb) I += sb.amp * Math.sin(2 * Math.PI * sb.f * t + sb.ph) * scanGate;
      const inc = inEdges[i];
      for (let k = 0; k < inc.length; k++) I += inc[k][1] * phi[inc[k][0]];
      I += noise[i];
      x[i] += (dt / tau[i]) * (I - x[i]);
      if (!Number.isFinite(x[i])) x[i] = 0; // safety net; params keep this far from firing
      // OU noise update
      noise[i] += dt * (-noise[i] / 0.8) + nsig[i] * Math.sqrt(dt) * (this.rng() * 2 - 1) * 1.73;
    }

    // --- decode commands from the descending population ---
    const meanOf = (name) => {
      const [a, b] = G[name];
      let s = 0; for (let i = a; i < b; i++) s += Math.tanh(x[i]);
      return s / (b - a);
    };
    const meanW = meanOf('W'), meanS = meanOf('S');
    const dw = meanOf('DW'), ds = meanOf('DS');
    const dl = meanOf('DL'), dr = meanOf('DR');
    const walkDrive = clamp(dw - ds, 0, 1);
    const vx = 0.25 * clamp((walkDrive - 0.1) / 0.5, 0, 1);
    const wz = 0.35 * clamp(dl - dr, -1, 1) * clamp(walkDrive * 1.5, 0, 1);
    const scanAmp = 0.3 + 0.7 * clamp(1 - walkDrive, 0, 1);
    const head = this.cmds.head;
    head.head_yaw = 0.45 * meanOf('SY') * scanAmp;
    head.head_pitch = 0.2 * meanOf('SP1') * scanAmp;
    head.head_roll = 0.25 * meanOf('SP2') * scanAmp;
    head.neck_pitch = 0.15 * meanOf('SP1') * scanAmp;
    this.cmds.vx = vx;
    this.cmds.wz = wz;
    this.cmds.walkDrive = walkDrive;
    this.cmds.mode = walkDrive > 0.3 ? 'walk' : 'stand';
    this.walkDrive = walkDrive;
    this.scanGate = 0.3 + 0.7 * clamp(1 - walkDrive, 0, 1);

    // --- homeostats: slow bout pressure gating the flip-flop circuits ---
    this.hw += dt * ((walkDrive > 0.3 ? HOMEOSTAT.hw[0] : 0) - HOMEOSTAT.hw[1] * this.hw);
    this.hs += dt * ((walkDrive <= 0.3 ? HOMEOSTAT.hs[0] : 0) - HOMEOSTAT.hs[1] * this.hs);
    this.hl += dt * ((dl - dr > 0.25 ? HOMEOSTAT.hl[0] : 0) - HOMEOSTAT.hl[1] * this.hl);
    this.hr += dt * ((dr - dl > 0.25 ? HOMEOSTAT.hr[0] : 0) - HOMEOSTAT.hr[1] * this.hr);

    // --- rendered activity ---
    for (let i = 0; i < N; i++) {
      const target = Math.tanh(x[i]);
      this.activity[i] += (target - this.activity[i]) * Math.min(1, dt * 8);
    }
  }

  setDynamics(on) {
    this.edgeLines.visible = on;
    this.pulses.visible = on;
  }

  // ---------- rendering ----------

  buildScene() {
    this.group = new THREE.Group();
    this.group.name = 'cns';

    // skeletons (one LineSegments draw call, per-vertex color updated live)
    const posArr = [];
    const colArr = [];
    this.vertRange = []; // neuron index -> [start, count) in vertices
    for (const n of this.neurons) {
      const start = posArr.length / 3;
      for (const p of n.segs) posArr.push(p.x, p.y, p.z);
      this.vertRange.push([start, n.segs.length]);
      for (let k = 0; k < n.segs.length; k++) colArr.push(0.2, 0.6, 0.6);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(posArr, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colArr, 3));
    this.skelMat = new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.85,
      blending: THREE.AdditiveBlending, depthWrite: false,
    });
    this.skeletons = new THREE.LineSegments(geo, this.skelMat);
    this.group.add(this.skeletons);

    // somas
    const sp = [], sc = [];
    this.neurons.forEach((n) => { sp.push(n.soma.x, n.soma.y, n.soma.z); sc.push(1, 1, 1); });
    const sgeo = new THREE.BufferGeometry();
    sgeo.setAttribute('position', new THREE.Float32BufferAttribute(sp, 3));
    sgeo.setAttribute('color', new THREE.Float32BufferAttribute(sc, 3));
    this.somaMat = new THREE.PointsMaterial({
      size: 9, vertexColors: true, transparent: true, opacity: 0.95,
      blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
    });
    this.somas = new THREE.Points(sgeo, this.somaMat);
    this.group.add(this.somas);

    // gray "obfuscated" blobs
    this.blobs = new THREE.Group();
    const blobMat = new THREE.MeshBasicMaterial({
      color: 0x5a6673, transparent: true, opacity: 0.13, depthWrite: false,
    });
    const blobSpecs = [
      [60, 380, 0, 250, 160, 120], [30, 300, 0, 150, 90, 100],
      [20, 30, 0, 110, 130, 80], [10, -180, 0, 95, 130, 75],
      [0, -330, 0, 80, 110, 60], [-10, -470, 0, 60, 90, 50],
      [40, 180, 0, 80, 60, 70], [-40, -60, 0, 70, 80, 60],
      [90, 250, 0, 90, 70, 60],
    ];
    for (const [x, y, z, rx, ry, rz] of blobSpecs) {
      const g = new THREE.IcosahedronGeometry(1, 2);
      const attr = g.attributes.position;
      for (let i = 0; i < attr.count; i++) {
        const s = 0.82 + this.rng() * 0.36;
        attr.setXYZ(i, attr.getX(i) * s, attr.getY(i) * s, attr.getZ(i) * s);
      }
      g.computeVertexNormals();
      const m = new THREE.Mesh(g, blobMat);
      m.position.set(x, y, z);
      m.scale.set(rx, ry, rz);
      m.rotation.y = this.rng() * Math.PI;
      this.blobs.add(m);
    }
    this.group.add(this.blobs);

    // connection overlay (connectome dynamics mode)
    const ep = [];
    for (const [a, b] of this.edges) {
      const na = this.neurons[a].soma, nb = this.neurons[b].soma;
      ep.push(na.x, na.y, na.z, nb.x, nb.y, nb.z);
    }
    const egeo = new THREE.BufferGeometry();
    egeo.setAttribute('position', new THREE.Float32BufferAttribute(ep, 3));
    this.edgeMat = new THREE.LineBasicMaterial({
      color: 0x2e6a72, transparent: true, opacity: 0.16,
      blending: THREE.AdditiveBlending, depthWrite: false,
    });
    this.edgeLines = new THREE.LineSegments(egeo, this.edgeMat);
    this.edgeLines.visible = false;
    this.group.add(this.edgeLines);

    // traveling pulse particles
    const PULSES = 90;
    this.pulseData = [];
    for (let i = 0; i < PULSES; i++) this.pulseData.push({ e: Math.floor(this.rng() * this.edges.length), t: this.rng(), sp: 0.6 + this.rng() * 1.8 });
    const pgeo = new THREE.BufferGeometry();
    pgeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(PULSES * 3), 3));
    this.pulseMat = new THREE.PointsMaterial({
      size: 6, color: 0x9ff2ec, transparent: true, opacity: 0.9,
      blending: THREE.AdditiveBlending, depthWrite: false,
    });
    this.pulses = new THREE.Points(pgeo, this.pulseMat);
    this.pulses.visible = false;
    this.group.add(this.pulses);

    this.bounds = { min: new THREE.Vector3(-220, -640, -160), max: new THREE.Vector3(260, 500, 160) };
  }

  // ---------- per-frame update: dynamics + paint ----------
  // drives: {jointName: [-1..1]} signed muscle states (readout);
  // sensory: {speed, yawRate, upright, fallen, headSpeed}
  // returns decoded commands {vx, wz, head, mode, walkDrive}
  update({ drives, sensory }, dt) {
    this.step(sensory ?? { speed: 0, yawRate: 0, upright: 1, fallen: 0, headSpeed: 0 }, drives ?? {}, Math.min(dt, 0.05));

    // paint skeletons from the simulated activity
    const col = this.skeletons.geometry.attributes.color;
    for (let i = 0; i < this.neurons.length; i++) {
      const a = this.activity[i];
      const mag = Math.min(1, Math.abs(a));
      // floor lifted from 0.22 → 0.45 so quiet neurons are still clearly
      // readable; the 0.55 slope keeps the active-vs-quiet contrast intact
      const bright = 0.45 + 0.55 * mag;
      const c = a >= 0 ? CYAN : ORANGE;
      const dim = a >= 0 ? CYAN_DIM : ORANGE_DIM;
      n_color.copy(dim).lerp(c, bright);
      const [start, count] = this.vertRange[i];
      for (let k = 0; k < count; k++) col.setXYZ(start + k, n_color.r, n_color.g, n_color.b);
    }
    col.needsUpdate = true;

    const scol = this.somas.geometry.attributes.color;
    for (let i = 0; i < this.neurons.length; i++) {
      const a = this.activity[i];
      const bright = 0.50 + 0.50 * Math.min(1, Math.abs(a));
      n_color.copy(a >= 0 ? CYAN : ORANGE).multiplyScalar(bright);
      scol.setXYZ(i, n_color.r, n_color.g, n_color.b);
    }
    scol.needsUpdate = true;

    // pulses: particles travel along connections whose source is actually firing
    if (this.pulses.visible) {
      const ppos = this.pulses.geometry.attributes.position;
      for (let i = 0; i < this.pulseData.length; i++) {
        const p = this.pulseData[i];
        p.t += p.sp * dt;
        if (p.t >= 1) {
          p.t = 0;
          // prefer edges from active neurons so traffic follows real activity
          for (let tries = 0; tries < 4; tries++) {
            const e = Math.floor(this.rng() * this.edges.length);
            const [s] = this.edges[e];
            if (Math.abs(this.activity[s]) > 0.25 || tries === 3) { p.e = e; break; }
          }
        }
        const na = this.neurons[this.edges[p.e][0]].soma;
        const nb = this.neurons[this.edges[p.e][1]].soma;
        ppos.setXYZ(i, na.x + (nb.x - na.x) * p.t, na.y + (nb.y - na.y) * p.t, na.z + (nb.z - na.z) * p.t);
      }
      ppos.needsUpdate = true;
    }
    return this.cmds;
  }
}

const n_color = new THREE.Color();
