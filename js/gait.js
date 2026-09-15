// gait.js — the "trained controller": foot-placement walking with numeric IK
// over the real MJCF joint chain, gaze-stabilized head with exploratory scans,
// plus an ablation mode ("untrained") that drives joints with smooth noise.
// Also derives per-joint signed muscle drives (the signal the CNS panel renders).
import * as THREE from 'three';

export const JOINT_ORDER = [
  'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
  'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
  'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle',
];
const LEG_JOINTS = ['hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle'];

const _p = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _ax = new THREE.Vector3();
const _v = new THREE.Vector3();
const _f = new THREE.Vector3();
const _t = new THREE.Vector3();
const _t2 = new THREE.Vector3();

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class GaitController {
  constructor(robot) {
    this.robot = robot;
    this.trained = true;
    this.physics = true;

    this.speed = 0.055;              // m/s forward
    this.omega = 2 * Math.PI * 1.45; // step cycle rad/s
    this.duty = 0.62;                // stance fraction of the cycle
    this.lift = 0.02;

    this.t = 0;
    this.heading = 0;
    this.pos = new THREE.Vector3(0, 0, 0);
    this.bodyRoll = 0; this.bodyPitch = 0; this.bodyHeight = 0;
    this.dip = 0;                    // physics: landing dip of the trunk

    this.scanYaw = 0; this.scanTarget = 0; this.scanTimer = 0;
    this.eventDuration = 0;
    this.rng = mulberry32(20260915);
    this.noise = this.makeNoise(this.rng);

    this.angles = {}; JOINT_ORDER.forEach((j) => (this.angles[j] = 0));
    this.drives = {}; JOINT_ORDER.forEach((j) => (this.drives[j] = 0));
    this.prevAngles = {};

    this.feet = {
      left: { phase: 0, plant: new THREE.Vector3(), swinging: false, from: new THREE.Vector3(), contact: true },
      right: { phase: Math.PI, plant: new THREE.Vector3(), swinging: false, from: new THREE.Vector3(), contact: true },
    };

    this.captureChains();
    this.placeFeetInitially();
  }

  // Per-leg data measured from the built scene graph at zero pose (MJCF quats
  // make hand-derived offsets unreliable, so we measure instead).
  captureChains() {
    this.chains = {};
    this.root = this.robot.root;
    this.root.updateMatrixWorld(true);

    for (const side of ['left', 'right']) {
      const bodyOf = (j) => {
        for (const [, rec] of this.robot.byBody) if (rec.joint && rec.joint.name === j) return rec;
        return null;
      };
      const recs = LEG_JOINTS.map((k) => bodyOf(`${side}_${k}`));
      const ankleRec = recs[4];
      const hipRec = recs[2]; // hip_pitch body = upper_leg_{side}

      const soleMesh = ankleRec.group.children.find(
        (c) => c.isMesh && c.userData.file === `sole_${side}`
      );
      // The sole STL's origin is NOT on the sole surface. Use the mesh's world
      // bounding-box bottom center as the true ground-contact point, expressed
      // in ankle-local coords so it rides with the foot.
      let tipLocal = new THREE.Vector3(0, 0, -0.064);
      if (soleMesh) {
        const box = new THREE.Box3().setFromObject(soleMesh);
        const soleTipWorld = box.getCenter(new THREE.Vector3());
        soleTipWorld.y = box.min.y;
        tipLocal = ankleRec.group.worldToLocal(soleTipWorld.clone());
      }

      // Anchors measured in WORLD coords at zero pose (root sits at the origin,
      // so world == wrapper-relative three-coords). MJCF quats make hand
      // conversion unreliable; measurements keep us honest.
      const hipAnchor = new THREE.Vector3();
      hipRec.group.getWorldPosition(hipAnchor);

      const kneeAnchor = new THREE.Vector3();
      recs[3].group.getWorldPosition(kneeAnchor);

      const ankleAnchor = new THREE.Vector3();
      ankleRec.group.getWorldPosition(ankleAnchor);

      const L1 = kneeAnchor.distanceTo(hipAnchor);
      const L2 = ankleAnchor.distanceTo(kneeAnchor);
      const footDrop = tipLocal.length();

      this.chains[side] = {
        joints: recs.map((r) => r.joint),
        ankleGroup: ankleRec.group,
        tipLocal,
        hipAnchor,
        L1, L2, footDrop,
      };
    }
    this.measureStance();
  }

  // The MJCF zero pose IS the natural standing pose: feet already rest at the
  // ground plane when the trunk wrapper sits at y≈0. Measure the natural foot
  // contact offsets and the standing height from that pose — no IK search.
  measureStance() {
    this.root.updateMatrixWorld(true);
    let hoverSum = 0;
    for (const side of ['left', 'right']) {
      const ch = this.chains[side];
      const tipWorld = ch.ankleGroup.localToWorld(ch.tipLocal.clone());
      ch.stance = { x: tipWorld.x, z: tipWorld.z };
      // raise the trunk 12 mm above natural contact: legs near-straight with a
      // soft knee flex (the -tipY0 exact value forced a deep squat)
      hoverSum += -tipWorld.y + 0.012;
    }
    this.chains.left.hover = hoverSum / 2;
    this.chains.right.hover = hoverSum / 2;
  }

  placeFeetInitially() {
    this.root.updateMatrixWorld(true);
    for (const side of ['left', 'right']) {
      const ch = this.chains[side];
      ch.ankleGroup.getWorldPosition(_p);
      this.feet[side].plant.set(_p.x, 0, _p.z);
      this.feet[side].from.copy(this.feet[side].plant);
    }
  }

  makeNoise(rng) {
    const out = {};
    for (const j of JOINT_ORDER) {
      const comps = [];
      for (let k = 0; k < 3; k++) {
        comps.push({ f: 0.6 + rng() * 2.4, p: rng() * Math.PI * 2, a: 0.4 + rng() * 0.6 });
      }
      out[j] = comps;
    }
    return out;
  }

  headingCurve(t) {
    return 0.28 * Math.sin(0.11 * t) + 0.12 * Math.sin(0.031 * t + 2.1);
  }

  // heading θ rotates the body about three +Y; the mesh faces +X at θ=0 and
  // the robot's left is -Z at θ=0 (MJCF +X/+Y after the z-up conversion).
  forwardOf(heading, out) {
    return out.set(Math.cos(heading), 0, -Math.sin(heading));
  }
  leftOf(heading, out) {
    return out.set(-Math.sin(heading), 0, -Math.cos(heading));
  }

  footTargetWorld(side, phase, out) {
    const fwd = this.forwardOf(this.heading, _t);
    const left = this.leftOf(this.heading, _t2);
    const ch = this.chains[side];
    // natural stance offset (measured at zero pose) rotated by heading
    const nomX = this.pos.x + fwd.x * ch.stance.x + left.x * ch.stance.z;
    const nomZ = this.pos.z + fwd.z * ch.stance.x + left.z * ch.stance.z;

    const f = ((phase % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    const cyc = f / (2 * Math.PI);
    if (cyc < this.duty) {
      out.copy(this.feet[side].plant);
      out.y = 0;
      this.feet[side].swinging = false;
    } else {
      const s = (cyc - this.duty) / (1 - this.duty);
      if (!this.feet[side].swinging) {
        this.feet[side].swinging = true;
        this.feet[side].from.copy(this.feet[side].plant);
        const lead = this.speed * (2 * Math.PI / this.omega) * this.duty * 0.5;
        this.feet[side].plant.set(nomX + fwd.x * lead, 0, nomZ + fwd.z * lead);
      }
      const e = s * s * (3 - 2 * s);
      out.lerpVectors(this.feet[side].from, this.feet[side].plant, e);
      out.y = this.lift * Math.pow(Math.sin(Math.PI * s), 0.9);
    }
    return out;
  }

  // Small-step heavily damped tracking. Gait targets move ~1-2 cm per frame,
  // so a few short DLS steps with strong damping (λ=2e-3, steps ≤0.12 rad,
  // joints held 4% inside their hard limits) track them smoothly — the
  // overshoot/divergence near the fully extended leg is impossible by design.
  solveLegIK(side, targetWorld) {
    const ch = this.chains[side];
    const joints = ch.joints;
    const lam = 2e-3;
    for (let iter = 0; iter < 6; iter++) {
      this.root.updateMatrixWorld(true);
      ch.ankleGroup.getWorldPosition(_p);
      const ex = targetWorld.x - _p.x, ey = targetWorld.y - _p.y, ez = targetWorld.z - _p.z;
      if (!Number.isFinite(ex + ey + ez)) break;
      if (Math.abs(ex) + Math.abs(ey) + Math.abs(ez) < 1e-4) break;

      const J = [new Float64Array(5), new Float64Array(5), new Float64Array(5)];
      for (let i = 0; i < 5; i++) {
        const g = joints[i].group;
        g.getWorldPosition(_t2);
        _ax.copy(joints[i].axis).applyQuaternion(g.getWorldQuaternion(_q));
        _v.copy(_p).sub(_t2);
        J[0][i] = _ax.y * _v.z - _ax.z * _v.y;
        J[1][i] = _ax.z * _v.x - _ax.x * _v.z;
        J[2][i] = _ax.x * _v.y - _ax.y * _v.x;
      }
      const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
        let s = 0; for (let i = 0; i < 5; i++) s += J[r][i] * J[c][i];
        A[r][c] = s + (r === c ? lam : 0);
      }
      const rhs = [ex, ey, ez];
      const y = solve3(A, rhs);
      if (!y.every(Number.isFinite)) break;
      for (let i = 0; i < 5; i++) {
        let d = 0; for (let r = 0; r < 3; r++) d += J[r][i] * y[r];
        d = THREE.MathUtils.clamp(d, -0.12, 0.12);
        const j = joints[i];
        j.angle = THREE.MathUtils.clamp(
          j.angle + d,
          j.range[0] * 0.96 - 0.002, j.range[1] * 0.96 + 0.002
        );
        this.robot.setJoint(j.name, j.angle);
        this.angles[j.name] = j.angle;
      }
    }
  }

  updateHead(dt, bodyRoll, bodyPitch) {
    this.scanTimer -= dt;
    if (this.scanTimer <= 0) {
      this.scanTarget = (this.rng() * 2 - 1) * 0.9;
      this.scanTimer = 1.4 + this.rng() * 2.2;
      this.eventDuration = 0;
    }
    this.eventDuration += dt;
    this.scanYaw += (this.scanTarget - this.scanYaw) * Math.min(1, dt * 3.2);
    this.setJ('head_yaw', this.scanYaw);
    this.setJ('neck_pitch', -bodyPitch * 0.85);
    this.setJ('head_pitch', -bodyPitch * 0.45 + 0.02 * Math.sin(2 * this.omega * this.t));
    this.setJ('head_roll', -bodyRoll * 0.7);
  }

  setJ(name, angle) {
    this.robot.setJoint(name, angle);
    this.angles[name] = this.robot.joints.get(name).angle;
  }

  update(dt) {
    dt = Math.min(dt, 0.05);
    this.t += dt;
    const root = this.robot.root;

    if (!this.trained) {
      for (const j of JOINT_ORDER) {
        let a = 0;
        for (const c of this.noise[j]) a += c.a * Math.sin(2 * Math.PI * c.f * this.t + c.p);
        const range = this.robot.joints.get(j).range;
        const amp = (Math.abs(range[0]) + Math.abs(range[1])) * 0.3;
        this.setJ(j, a * amp);
      }
      root.position.set(0, 0, 0);
      root.rotation.set(-Math.PI / 2, 0, 0.03 * Math.sin(2.1 * this.t), 'YXZ');
      this.drivesFromVelocity(dt, 1.6);
      this.bodyHeight = 0; this.bodyRoll = 0; this.bodyPitch = 0;
      return this.snapshot();
    }

    const targetHeading = this.headingCurve(this.t);
    this.heading = targetHeading;
    const fwd = this.forwardOf(this.heading, _t);
    this.pos.x += fwd.x * this.speed * dt;
    this.pos.z += fwd.z * this.speed * dt;

    const bob = 0.004 * Math.sin(2 * this.omega * this.t);
    this.bodyRoll = 0.035 * Math.sin(this.omega * this.t);
    this.bodyPitch = 0.03 * Math.sin(2 * this.omega * this.t + 0.7);
    let h = this.chains.left.hover;
    if (this.physics) {
      this.dip += (0 - this.dip) * Math.min(1, dt * 3);
      h -= this.dip;
    }
    this.bodyHeight = h;
    root.position.set(this.pos.x, h + bob, this.pos.z);
    root.rotation.set(-Math.PI / 2 + this.bodyPitch, this.heading, this.bodyRoll, 'YXZ');

    for (const side of ['left', 'right']) {
      const ph = this.omega * this.t + this.feet[side].phase;
      const before = this.feet[side].contact;
      const target = this.footTargetWorld(side, ph, new THREE.Vector3());
      this.solveLegIK(side, target);
      this.feet[side].contact = !this.feet[side].swinging;
      if (this.physics && this.feet[side].contact && !before) {
        this.dip = Math.min(this.dip + 0.006, 0.012);
      }
    }
    this.updateHead(dt, this.bodyRoll, this.bodyPitch);
    this.drivesFromVelocity(dt, 1);
    return this.snapshot();
  }

  drivesFromVelocity(dt, gain) {
    for (const j of JOINT_ORDER) {
      const prev = this.prevAngles[j] ?? this.angles[j];
      const range = this.robot.joints.get(j).range;
      const scale = Math.max(1e-4, (range[1] - range[0]) / 2);
      const vel = (this.angles[j] - prev) / Math.max(dt, 1e-4);
      const d = THREE.MathUtils.clamp((vel / scale) * 0.35 * gain, -1, 1);
      this.drives[j] += (d - this.drives[j]) * Math.min(1, dt * 10);
      this.prevAngles[j] = this.angles[j];
    }
  }

  snapshot() {
    return {
      t: this.t,
      angles: { ...this.angles },
      drives: { ...this.drives },
      body: {
        heading: this.heading,
        roll: this.bodyRoll,
        pitch: this.bodyPitch,
        height: this.bodyHeight,
        speed: this.trained ? this.speed : 0,
        pos: this.pos.clone(),
      },
      scanYaw: this.scanYaw,
      eventDuration: this.eventDuration,
    };
  }
}

function solve3(A, b) {
  const n = 3;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-12) continue;
    [M[col], M[piv]] = [M[piv], M[col]];
    const d = M[col][col];
    for (let c = col; c <= n; c++) M[col][c] /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      if (!f) continue;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row) => row[n]);
}
