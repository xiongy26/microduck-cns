#!/usr/bin/env node
// maze_sim.mjs — 2-D closed-loop maze harness for the connectome brain.
//
// Runs the same 200-neuron CTRNN that drives the robot in the browser against
// a kinematic ground robot (first-order velocity/turn-rate lags) plus the same
// ray-AABB vision model, inside the MAZE_WALLS layout from js/physics.js.
// Useful for tuning circuits without waiting on MuJoCo/ONNX: it needs no
// browser, WASM or network.
//
// Usage:  node tools/maze_sim.mjs [seconds-per-run] [run-count]
//
// js/connectome.js imports the bare specifier 'three', which Node cannot
// resolve, so this script copies it to a temp file and rewrites the import to
// the vendored three.module.js before loading it.
import { mkdtempSync, copyFileSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const T = Number(process.argv[2] ?? 180);
const RUNS = Number(process.argv[3] ?? 20);

const DT = 1 / 30;
const FAR = 1.3, NEAR = 0.25;
const RAY_YAW = [-0.95, -0.5, 0, 0.5, 0.95];
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

// MAZE_WALLS from js/physics.js, flattened to 2-D {c:[x,y], h:[hx,hy]}
const wallsSrc = readFileSync(join(ROOT, 'js/physics.js'), 'utf8');
const wallsBlock = wallsSrc.slice(wallsSrc.indexOf('export const MAZE_WALLS'));
const walls = [...wallsBlock.matchAll(/pos: \[([-\d., ]+)\],\s+size: \[([-\d., ]+)\]/g)]
  .map((m) => {
    const [cx, cy] = m[1].split(',').map(Number);
    const [hx, hy] = m[2].split(',').map(Number);
    return { c: [cx, cy], h: [hx, hy] };
  });
if (walls.length === 0) throw new Error('no MAZE_WALLS parsed from js/physics.js');

function rayDist(px, py, dx, dy, w) {
  let tmin = -Infinity, tmax = Infinity;
  for (const [p, d, c, h] of [[px, dx, w.c[0], w.h[0]], [py, dy, w.c[1], w.h[1]]]) {
    if (Math.abs(d) < 1e-9) { if (p < c - h || p > c + h) return Infinity; }
    else {
      let t1 = (c - h - p) / d, t2 = (c + h - p) / d;
      if (t1 > t2) [t1, t2] = [t2, t1];
      tmin = Math.max(tmin, t1); tmax = Math.min(tmax, t2);
    }
  }
  if (tmax < Math.max(tmin, 0)) return Infinity;
  return tmin > 0 ? tmin : Infinity;
}

function rayVis(px, py, th) {
  let vL = 0, vR = 0;
  for (const ry of RAY_YAW) {
    const dx = Math.cos(th + ry), dy = Math.sin(th + ry);
    let d = FAR;
    for (const w of walls) d = Math.min(d, rayDist(px, py, dx, dy, w));
    const inten = clamp((FAR - Math.min(d, FAR)) / (FAR - NEAR), 0, 1);
    if (ry >= 0) vL = Math.max(vL, inten); else vR = Math.max(vR, inten);
  }
  return { vL, vR };
}

const exited = (x, y) => y > 4.5 || (y > 3.5 && x < 0.6);

// load Connectome with the three import rewritten
const tmp = mkdtempSync(join(tmpdir(), 'maze-sim-'));
const connPath = join(tmp, 'connectome.mjs');
let src = readFileSync(join(ROOT, 'js/connectome.js'), 'utf8');
src = src.replace("from 'three'", `from '${pathToFileURL(join(ROOT, 'assets/vendor/three.module.js')).href}'`);
writeFileSync(connPath, src);
const { Connectome } = await import(pathToFileURL(connPath).href);

function runMaze(seed, T = 180) {
  const cns = new Connectome(seed);
  let x = 0, y = 0, th = 0, v = 0, om = 0, t = 0, contacts = 0;
  const N = Math.round(T / DT);
  for (let i = 0; i < N; i++) {
    const { vL, vR } = rayVis(x, y, th);
    cns.step({ speed: Math.abs(v), yawRate: om, upright: 1, fallen: 0, headSpeed: 0, visL: vL, visR: vR }, {}, DT);
    const { vx, wz } = cns.cmds;
    v += (vx - v) * Math.min(1, DT * 1.2);
    om += (wz - om) * Math.min(1, DT * 2.5);
    const px = x, py = y;
    x += v * Math.cos(th) * DT;
    y += v * Math.sin(th) * DT;
    th += om * DT;
    let hit = false;
    for (const w of walls) {
      if (Math.abs(x - w.c[0]) < w.h[0] + 0.18 && Math.abs(y - w.c[1]) < w.h[1] + 0.18) { hit = true; break; }
    }
    if (hit) { x = px; y = py; v = 0; contacts++; }
    t += DT;
    if (exited(x, y)) return { escaped: true, t, contacts };
  }
  return { escaped: false, t, contacts };
}

let ok = 0;
const times = [];
let contacts = 0;
for (let k = 0; k < RUNS; k++) {
  const r = runMaze(1000 + k * 137);
  if (r.escaped) { ok++; times.push(r.t); }
  contacts += r.contacts;
}
times.sort((a, b) => a - b);
const med = times.length ? times[Math.floor(times.length / 2)].toFixed(0) : '-';
console.log(`maze escape: ${ok}/${RUNS} within ${T}s | median ${med}s | fastest ${times[0]?.toFixed(0) ?? '-'}s | wall contacts ${contacts}`);
