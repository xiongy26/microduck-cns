// robot.js — rebuilds the official Pollen Robotics MicroDuck kinematic tree
// (converted from microduck_rl MJCF by tools/mjcf_to_json.py) in Three.js.
import * as THREE from 'three';
import { STLLoader } from 'three/addons/STLLoader.js';

// Material palette matched to the real robot (white shells, orange feet,
// dark Dynamixel XL330 servos, metal bearings).
const MATERIALS = {
  shell:      () => new THREE.MeshStandardMaterial({ color: 0xe3e6e8, roughness: 0.45, metalness: 0.05 }),
  lightGray:  () => new THREE.MeshStandardMaterial({ color: 0xcdd2d6, roughness: 0.55, metalness: 0.1 }),
  dark:       () => new THREE.MeshStandardMaterial({ color: 0x39424e, roughness: 0.6,  metalness: 0.2 }),
  orange:     () => new THREE.MeshStandardMaterial({ color: 0xf08a2c, roughness: 0.6,  metalness: 0.05 }),
  metal:      () => new THREE.MeshStandardMaterial({ color: 0x9aa3ad, roughness: 0.35, metalness: 0.7 }),
  lens:       () => new THREE.MeshStandardMaterial({ color: 0x14181d, roughness: 0.2,  metalness: 0.4 }),
  pcb:        () => new THREE.MeshStandardMaterial({ color: 0x2f7d4f, roughness: 0.7,  metalness: 0.1 }),
};

const MESH_MATERIAL = {
  trunk_base: 'shell', right_shell: 'shell', left_shell: 'shell',
  top_head_shell: 'shell', bottom_head_shell: 'shell', face_part: 'shell',
  jaw_soft: 'shell', jaw: 'shell', soft_mouth_top: 'shell',
  noenoeil: 'lens', lens: 'lens', np_f970: 'lens',
  foot_left: 'orange', foot_right: 'orange', sole_left: 'orange', sole_right: 'orange',
  xl330: 'dark', power_support: 'dark', motor_support: 'dark', speaker: 'dark',
  seeed_bearing__configuration__22x16x4: 'metal',
  seeed_bearing__configuration_default: 'metal', bearing_roll: 'metal',
  banana_pcb_locker: 'pcb', pcb__raspberry_pi_zero_2_w: 'pcb', elec_rpi_robot_hat_pcb: 'pcb',
  yaw2roll: 'lightGray', hip_l: 'lightGray', upper_leg_left: 'lightGray',
  upper_leg_right: 'lightGray', upper_leg_rigidity_plate: 'lightGray',
  leg: 'lightGray', ankle_left: 'lightGray', ankle_right: 'lightGray',
  neck: 'lightGray', neck_pitch: 'lightGray', yaw_roll_motion: 'lightGray',
  m12_lens_holder: 'dark',
};

const _q = new THREE.Quaternion();
const _axis = new THREE.Vector3();

export async function buildRobot(url = 'assets/robot.json', meshDir = 'assets/meshes/') {
  const data = await (await fetch(url)).json();
  const loader = new STLLoader();
  const geoCache = new Map();
  const loadGeo = async (file) => {
    if (!geoCache.has(file)) {
      const g = await loader.loadAsync(meshDir + file);
      g.computeBoundingSphere();
      geoCache.set(file, g);
    }
    return geoCache.get(file);
  };

  const groups = new Map();      // body name -> THREE.Group
  const joints = new Map();      // joint name -> {group, baseQuat, axis, range, angle}
  const byBody = new Map();      // body name -> record
  const root = new THREE.Group(); // wrapper used by locomotion
  root.name = 'robotRoot';

  // parents always appear before children in the MJCF document order
  for (const b of data.bodies) {
    const g = new THREE.Group();
    g.name = b.name;
    g.position.fromArray(b.pos);
    _q.set(b.quat[1], b.quat[2], b.quat[3], b.quat[0]);
    g.quaternion.copy(_q);
    const parent = b.parent ? groups.get(b.parent) : root;
    parent.add(g);
    groups.set(b.name, g);

    const rec = { name: b.name, group: g, joint: null };
    if (b.joint) {
      const baseQuat = g.quaternion.clone();
      const j = {
        name: b.joint.name, group: g, baseQuat,
        axis: new THREE.Vector3().fromArray(b.joint.axis).normalize(),
        range: b.joint.range, angle: 0,
      };
      joints.set(j.name, j);
      rec.joint = j;
    }
    byBody.set(b.name, rec);

    for (const m of b.meshes) {
      const geo = await loadGeo(m.file);
      const mesh = new THREE.Mesh(geo, makeMaterial(m.file));
      mesh.position.fromArray(m.pos);
      _q.set(m.quat[1], m.quat[2], m.quat[3], m.quat[0]);
      mesh.quaternion.copy(_q);
      mesh.userData.file = m.file.replace(/\.stl$/, '');
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      g.add(mesh);
    }
  }

  const materials = new Set();
  groups.forEach((g) => g.traverse((o) => { if (o.isMesh) materials.add(o.material); }));

  function setJoint(name, angle) {
    const j = joints.get(name);
    if (!j) return;
    j.angle = THREE.MathUtils.clamp(angle, j.range[0], j.range[1]);
    _axis.copy(j.axis);
    _q.setFromAxisAngle(_axis, j.angle);
    j.group.quaternion.copy(j.baseQuat).multiply(_q);
  }

  return {
    root, groups, joints, byBody, materials,
    trunk: groups.get('trunk_base'),
    setJoint,
    setAllWireframe(on) { materials.forEach((m) => { m.wireframe = on; }); },
  };
}

function makeMaterial(meshFile) {
  const key = meshFile.replace(/\.stl$/, '');
  const mat = MESH_MATERIAL[key] || 'lightGray';
  return MATERIALS[mat]();
}
