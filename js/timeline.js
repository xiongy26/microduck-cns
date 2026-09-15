// timeline.js — 14 joint tracks (a 20 s ring buffer of recorded controller
// output), canvas strip rendering, playhead + scrub.
import { JOINT_ORDER } from './gait.js';

const WINDOW_S = 20;
const FPS = 60;
const N = WINDOW_S * FPS;

export class Timeline {
  constructor(container) {
    this.order = JOINT_ORDER;
    this.ranges = null; // joint name -> half-range (for normalize)
    this.buf = new Float32Array(this.order.length * N);
    this.driveBuf = new Float32Array(this.order.length * N);
    this.writeIdx = 0;
    this.filled = 0;
    this.canvases = [];
    this.buildDOM(container);
  }

  buildDOM(container) {
    const labels = {
      left_hip_yaw: 'L hip yaw', left_hip_roll: 'L hip roll', left_hip_pitch: 'L hip pitch',
      left_knee: 'L knee', left_ankle: 'L ankle',
      neck_pitch: 'neck pitch', head_pitch: 'head pitch', head_yaw: 'head yaw', head_roll: 'head roll',
      right_hip_yaw: 'R hip yaw', right_hip_roll: 'R hip roll', right_hip_pitch: 'R hip pitch',
      right_knee: 'R knee', right_ankle: 'R ankle',
    };
    for (const j of this.order) {
      const div = document.createElement('div');
      div.className = 'track';
      const label = document.createElement('span');
      label.className = 'tlabel';
      label.textContent = labels[j] || j;
      const c = document.createElement('canvas');
      div.appendChild(label);
      div.appendChild(c);
      container.appendChild(div);
      this.canvases.push(c);
    }
  }

  setRanges(joints) {
    this.ranges = {};
    for (const j of this.order) {
      const r = joints.get(j).range;
      this.ranges[j] = Math.max(Math.abs(r[0]), Math.abs(r[1]));
    }
  }

  record(angles, drives) {
    for (let i = 0; i < this.order.length; i++) {
      const j = this.order[i];
      this.buf[i * N + this.writeIdx] = angles[j] / this.ranges[j];
      this.driveBuf[i * N + this.writeIdx] = drives[j];
    }
    this.writeIdx = (this.writeIdx + 1) % N;
    this.filled = Math.min(this.filled + 1, N);
  }

  // normalized joint angles at fractional position of the 20 s window
  sampleAt(frac, outAngles, outDrives) {
    const idx = Math.floor(frac * N) % N;
    for (let i = 0; i < this.order.length; i++) {
      outAngles[this.order[i]] = this.buf[i * N + idx] * this.ranges[this.order[i]];
      if (outDrives) outDrives[this.order[i]] = this.driveBuf[i * N + idx];
    }
    return outAngles;
  }

  draw() {
    const BARS = 46;
    for (let i = 0; i < this.order.length; i++) {
      const c = this.canvases[i];
      const w = c.clientWidth, h = c.clientHeight;
      if (c.width !== w * 2) { c.width = w * 2; c.height = h * 2; }
      const g = c.getContext('2d');
      g.setTransform(2, 0, 0, 2, 0, 0);
      g.clearRect(0, 0, w, h);
      const mid = h / 2;
      const bw = w / BARS;
      for (let b = 0; b < BARS; b++) {
        const idx = (this.writeIdx - BARS + b + N * 2) % N;
        const v = this.buf[i * N + idx];
        const bh = Math.max(2, Math.abs(v) * (h / 2 - 2));
        const fresh = b >= BARS - 3;
        g.fillStyle = fresh ? '#e8b47e' : '#c08d58';
        if (v >= 0) g.fillRect(b * bw + 0.5, mid - bh, Math.max(1.5, bw - 1), bh);
        else g.fillRect(b * bw + 0.5, mid, Math.max(1.5, bw - 1), bh);
      }
      g.fillStyle = 'rgba(150,170,190,0.35)';
      g.fillRect(0, mid - 0.5, w, 1);
      // playhead
      g.fillStyle = '#e8edf2';
      g.fillRect(w - 1.5, 0, 1.5, h);
    }
  }
}
