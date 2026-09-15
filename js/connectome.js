// connectome.js — a 192-neuron Male CNS subset rendered as anatomical skeletons.
// Layout follows real male fly anatomy: brain (supraesophageal ganglion),
// cervical connective, and a thoracic+abdominal ventral nerve cord with
// lateral nerve roots. Skeletons are procedurally grown, but the wiring
// counts (192 neurons / 2,456 connections) and the pool-to-joint mapping are
// the real contract of this demo: every VNC motor pool drives one MicroDuck
// joint, cyan = positive / orange = negative muscle state.
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

const CYAN = new THREE.Color('#3fd8d0');
const ORANGE = new THREE.Color('#ff8a3c');
const CYAN_DIM = new THREE.Color('#1b6b68');
const ORANGE_DIM = new THREE.Color('#8a4a20');

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

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

export class Connectome {
  constructor(seed = 19200301) {
    this.rng = mulberry32(seed);
    this.neurons = [];
    this.growAll();
    this.edges = this.wire();
    this.activity = new Float32Array(NEURON_COUNT);
    this.activityTarget = new Float32Array(NEURON_COUNT);
    this.buildScene();
  }

  // ---------- anatomy ----------

  growAll() {
    const rng = this.rng;
    const [b0, b1] = POPS.brain;
    for (let i = b0; i < b1; i++) this.neurons.push(this.growBrainNeuron());
    const [d0, d1] = POPS.descending;
    for (let i = d0; i < d1; i++) this.neurons.push(this.growDescendingNeuron());
    // motor pools, topographic along the cord
    let mi = POPS.motor[0];
    for (const pool of POOL_LAYOUT) {
      const yCenter = { pro: 10, meso: -105, meta: -215, sog: 120 }[pool.seg];
      for (let k = 0; k < pool.n; k++) {
        this.neurons.push(this.growMotorNeuron(yCenter + (rng() - 0.5) * 60, mi, pool.j));
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
      n.w = 0.45 + this.rng() * 0.55;      // weight within pool
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

  // ---------- wiring: exactly CONNECTION_COUNT directed edges ----------

  wire() {
    const rng = this.rng;
    const pick = (pop) => {
      const [a, b] = POPS[pop];
      return a + Math.floor(rng() * (b - a));
    };
    const categories = [
      ['brain', 'brain', 520], ['brain', 'descending', 208],
      ['descending', 'motor', 864], ['motor', 'local', 96], ['local', 'motor', 96],
      ['motor', 'ascending', 240], ['ascending', 'brain', 432],
    ];
    const edges = [];
    for (const [src, dst, count] of categories) {
      for (let k = 0; k < count; k++) edges.push([pick(src), pick(dst)]);
    }
    // exact total guard (categories sum to 2456 already; assert in dev)
    return edges;
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

  setDynamics(on) {
    this.edgeLines.visible = on;
    this.pulses.visible = on;
  }

  // ---------- per-frame activity ----------
  // drives: {jointName: [-1..1]} signed muscle states; signals: {turn, scan, speed}

  update({ drives, signals }, dt) {
    const t = performance.now() / 1000;
    for (let i = 0; i < this.neurons.length; i++) {
      const n = this.neurons[i];
      let target;
      const wob = 0.22 * Math.sin(t * n.wobF * 2 * Math.PI + n.wob);
      if (n.pool && drives[n.pool] !== undefined) {
        target = THREE.MathUtils.clamp(drives[n.pool] * n.w * 1.15 + wob * 0.5, -1, 1);
      } else if (n.pop === 'brain') {
        // command/association cells follow turn rate & scan events
        const cmd = signals.turn * 1.4 + signals.scan * 0.8 + Math.sin(t * 0.37 + i * 0.7) * 0.35;
        target = THREE.MathUtils.clamp(cmd * n.w + wob, -1, 1);
      } else if (n.pop === 'descending') {
        target = THREE.MathUtils.clamp((signals.turn * 1.2 + signals.speed * 2) * n.w + wob * 0.6, -1, 1);
      } else if (n.pop === 'ascending') {
        target = THREE.MathUtils.clamp(Math.sin(t * 2.9 + i) * signals.speed * 3 + wob * 0.5, -1, 1);
      } else {
        target = wob * 0.8;
      }
      this.activityTarget[i] = target;
      this.activity[i] += (target - this.activity[i]) * Math.min(1, dt * 8);
    }

    // paint skeletons
    const col = this.skeletons.geometry.attributes.color;
    for (let i = 0; i < this.neurons.length; i++) {
      const a = this.activity[i];
      const mag = Math.min(1, Math.abs(a));
      const bright = 0.22 + 0.78 * mag;
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
      const bright = 0.35 + 0.65 * Math.min(1, Math.abs(a));
      n_color.copy(a >= 0 ? CYAN : ORANGE).multiplyScalar(bright);
      scol.setXYZ(i, n_color.r, n_color.g, n_color.b);
    }
    scol.needsUpdate = true;

    // pulses
    if (this.pulses.visible) {
      const ppos = this.pulses.geometry.attributes.position;
      for (let i = 0; i < this.pulseData.length; i++) {
        const p = this.pulseData[i];
        p.t += p.sp * dt;
        if (p.t >= 1) {
          p.t = 0;
          p.e = Math.floor(Math.random() * this.edges.length);
        }
        const na = this.neurons[this.edges[p.e][0]].soma;
        const nb = this.neurons[this.edges[p.e][1]].soma;
        ppos.setXYZ(i, na.x + (nb.x - na.x) * p.t, na.y + (nb.y - na.y) * p.t, na.z + (nb.z - na.z) * p.t);
      }
      ppos.needsUpdate = true;
    }
  }
}

const n_color = new THREE.Color();
